//! A [`FileSource`] wrapper for scans that carry deletion vectors.
//!
//! `DeltaScanStream` consumes a file's keep mask positionally: the first rows
//! of the file take the head of the mask, and so on. The mask therefore lands
//! correctly only when the file arrives whole, in physical row order, with
//! every row still present. Three DataFusion optimisations break one of those:
//!
//! * byte-range splitting hands one file to several scan partitions, which
//!   `EnforceDistribution` then merges with a `CoalescePartitionsExec` to
//!   satisfy `DeltaScanExec`'s single-partition request -- merged, but not
//!   ordered,
//! * a pushed-down filter drops rows inside the Parquet reader, beneath the
//!   mask, so the surviving rows no longer align with it,
//! * sort pushdown lets the source hand back row groups in another order.
//!
//! `DeltaScanExec` declines the plan-level forms of all three. This closes the
//! source-level ones, which those guards do not reach: upstream #4692 uses
//! `FileScanConfigBuilder::with_output_partitioning` for the first, which
//! DataFusion 53 does not have.
//!
//! `WholeFileSource` implements DataFusion's [`FileSource`] trait, and every
//! method of that trait either forwards to the inner source or refuses. The
//! refusals are the three above. The forwarding needs more care, because some
//! `FileSource` methods do not answer a question about the source -- they hand
//! back a *replacement* for it, which DataFusion then swaps in:
//!
//! ```ignore
//! // datafusion-datasource/src/file_scan_config.rs
//! let source = self.file_source.with_batch_size(batch_size);
//! ```
//!
//! Forwarding such a method naively returns the bare inner source, and this
//! wrapper vanishes from the plan. That failure is silent: it compiles, and a
//! file too small to be split still reads correctly, so the guard is gone long
//! before any test notices. `with_batch_size` and `try_pushdown_projection`
//! are the two, and both re-wrap what they return.

use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::config::ConfigOptions;
use datafusion::datasource::physical_plan::{FileOpener, FileScanConfig, FileSource};
use datafusion::datasource::table_schema::TableSchema;
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::filter_pushdown::{FilterPushdownPropagation, PushedDown};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::sort_pushdown::SortOrderPushdownResult;
use datafusion::physical_plan::{DisplayFormatType, PhysicalExpr};
use object_store::ObjectStore;

/// Wrap `inner` so the scan keeps whole files, in order, with every row.
pub(super) fn keep_files_whole(inner: Arc<dyn FileSource>) -> Arc<dyn FileSource> {
    Arc::new(WholeFileSource { inner })
}

struct WholeFileSource {
    inner: Arc<dyn FileSource>,
}

impl FileSource for WholeFileSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Result<Arc<dyn FileOpener>> {
        self.inner
            .create_file_opener(object_store, base_config, partition)
    }

    /// Delegates, so anything downcasting to the concrete source still finds
    /// it. Nothing downcasts to this wrapper; `fmt_extra` is how a plan shows
    /// that it is installed.
    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }

    fn table_schema(&self) -> &TableSchema {
        self.inner.table_schema()
    }

    /// Re-wraps: `FileScanConfigBuilder::with_batch_size` replaces the source
    /// with whatever this returns, so handing back the bare inner source would
    /// drop the guard.
    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        keep_files_whole(self.inner.with_batch_size(batch_size))
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.inner.filter()
    }

    /// Delegated, not defaulted: the config derives the output schema from it,
    /// and reporting `None` for a source that does project would mis-describe
    /// the scan.
    fn projection(&self) -> Option<&ProjectionExprs> {
        self.inner.projection()
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        self.inner.metrics()
    }

    fn file_type(&self) -> &str {
        self.inner.file_type()
    }

    /// Marks the scan in `EXPLAIN` output, which is the only way to see the
    /// guard from outside; the tests assert on it.
    fn fmt_extra(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        self.inner.fmt_extra(t, f)?;
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, ", deletion_vector_row_order=preserved")
            }
            _ => Ok(()),
        }
    }

    fn supports_repartitioning(&self) -> bool {
        false
    }

    /// `repartitioned`'s default already returns `None` once
    /// `supports_repartitioning` is false. Refusing here too keeps the answer
    /// right if that default ever stops consulting it.
    fn repartitioned(
        &self,
        _target_partitions: usize,
        _repartition_file_min_size: usize,
        _output_ordering: Option<LexOrdering>,
        _config: &FileScanConfig,
    ) -> Result<Option<FileScanConfig>> {
        Ok(None)
    }

    /// A filter evaluated inside the reader drops rows before the mask sees
    /// them. The trait's default refuses as well; saying so here keeps the
    /// reason next to the other two refusals.
    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            vec![PushedDown::No; filters.len()],
        ))
    }

    /// `ParquetSource` implements this; sort pushdown may reorder row groups,
    /// which slides the mask against the rows.
    fn try_pushdown_sort(
        &self,
        _order: &[PhysicalSortExpr],
        _eq_properties: &EquivalenceProperties,
    ) -> Result<SortOrderPushdownResult<Arc<dyn FileSource>>> {
        Ok(SortOrderPushdownResult::Unsupported)
    }

    /// Allowed: a projection changes which columns are read, not which rows or
    /// their order. Re-wraps for the same reason as `with_batch_size`.
    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn FileSource>>> {
        Ok(self
            .inner
            .try_pushdown_projection(projection)?
            .map(keep_files_whole))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::datasource::physical_plan::ParquetSource;
    use datafusion::datasource::table_schema::TableSchema;
    use datafusion::physical_expr::projection::ProjectionExprs;
    use arrow_schema::{DataType, Field, Schema};

    fn wrapped() -> Arc<dyn FileSource> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("val", DataType::Utf8, true),
        ]));
        keep_files_whole(Arc::new(ParquetSource::new(TableSchema::from_file_schema(
            schema,
        ))))
    }

    /// The guard's whole job. `DeltaScanExec` asking for one input partition
    /// does not stop the source splitting a file, because the request is met
    /// with an unordered merge.
    #[test]
    fn refuses_to_be_split() {
        assert!(!wrapped().supports_repartitioning());
    }

    /// The failure this file exists to prevent, and the one nothing else
    /// catches: `FileScanConfigBuilder` assigns whatever `with_batch_size`
    /// returns, so a forward that does not re-wrap silently replaces the guard
    /// with the bare inner source. Everything still compiles and small files
    /// still read correctly, so only an assertion here notices.
    #[test]
    fn survives_with_batch_size() {
        assert!(
            !wrapped().with_batch_size(8192).supports_repartitioning(),
            "with_batch_size returned an unwrapped source; the guard is gone"
        );
    }

    /// Same trap on the other method that returns a replacement source.
    /// Projection pushdown is allowed -- it changes columns, not row order --
    /// but it must not cost the guard.
    #[test]
    fn survives_projection_pushdown() {
        let source = wrapped();
        if let Some(pushed) = source
            .try_pushdown_projection(&ProjectionExprs::from(vec![]))
            .unwrap()
        {
            assert!(
                !pushed.supports_repartitioning(),
                "try_pushdown_projection returned an unwrapped source; the guard is gone"
            );
        }
    }

    /// A filter evaluated in the reader drops rows before the mask is applied
    /// to them, so none may be accepted.
    #[test]
    fn accepts_no_filters() {
        let source = wrapped();
        let config = datafusion::common::config::ConfigOptions::default();
        let result = source.try_pushdown_filters(vec![], &config).unwrap();
        assert!(
            result.updated_node.is_none(),
            "the source took ownership of filters it must not evaluate"
        );
    }
}
