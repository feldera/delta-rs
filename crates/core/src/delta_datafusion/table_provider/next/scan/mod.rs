//! Kernel-based Delta table scanning with optimized query execution.
//!
//! This module provides efficient table scanning using Delta Kernel, integrating with
//! DataFusion's query engine. It supports:
//!
//! - **Physical scan execution** ([`DeltaScanExec`]) - Reads Parquet data files and applies
//!   Delta protocol transformations (column mapping, deletion vectors, partition values)
//! - **Metadata-only scans** ([`DeltaScanMetaExec`]) - Answers queries like `COUNT(*)`
//!   using file statistics without reading data files
//! - **Predicate pushdown** - Pushes filters to both kernel file skipping and Parquet readers
//!   for efficient data pruning
//! - **Multi-store support** - Handles files across different object stores in a single query
//!
//! The scan planning process in [`plan`] determines which files to read and how to apply
//! predicates, while execution plans handle the actual data reading and transformation.

use std::{collections::VecDeque, pin::Pin, sync::Arc};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_cast::{CastOptions, cast_with_options};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaBuilder, SchemaRef};
use chrono::{TimeZone as _, Utc};
use dashmap::DashMap;
use datafusion::{
    catalog::Session,
    common::{
        ColumnStatistics, HashMap, Result, Statistics, ToDFSchema, plan_err,
        stats::Precision,
        tree_node::{Transformed, TreeNode as _},
    },
    config::TableParquetOptions,
    datasource::physical_plan::{ParquetSource, parquet::CachedParquetFileReaderFactory},
    error::DataFusionError,
    execution::object_store::ObjectStoreUrl,
    logical_expr::ColumnarValue,
    physical_expr::{PhysicalExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan,
        empty::EmptyExec,
        metrics::{ExecutionPlanMetricsSet, MetricBuilder},
        union::UnionExec,
    },
    prelude::Expr,
};
use datafusion_datasource::{
    PartitionedFile, TableSchema, compute_all_files_statistics, file_groups::FileGroup,
    file_scan_config::FileScanConfigBuilder, source::DataSourceExec,
};
use datafusion_physical_expr_adapter::{
    BatchAdapter, BatchAdapterFactory, DefaultPhysicalExprAdapterFactory, PhysicalExprAdapter,
    PhysicalExprAdapterFactory,
};
use delta_kernel::{
    Engine, Expression, expressions::StructData, scan::ScanMetadata, table_features::TableFeature,
};
use futures::{Stream, TryStreamExt as _, try_join};
use itertools::Itertools as _;
use object_store::{ObjectMeta, path::Path};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use tokio::sync::mpsc;
use tracing::debug;
use url::Url;

use self::dv::{into_keep_masks, load_deletion_vectors};
pub use self::exec::DeltaScanExec;
use self::exec_meta::DeltaScanMetaExec;
pub(crate) use self::plan::{KernelScanPlan, ProjectedScanContract, supports_filters_pushdown};
use self::replay::{ScanFileContext, ScanFileStream};
use super::FileSelection;
use crate::{
    DeltaTableError,
    delta_datafusion::{
        DeltaScanConfig,
        engine::{AsObjectStoreUrl as _, to_datafusion_scalar},
        file_id::wrap_file_id_value,
        table_provider::next::DeletionVectorSelection,
    },
};

mod dv;
mod exec;
mod exec_meta;
mod plan;
mod replay;

type ScanMetadataStream = Pin<Box<dyn Stream<Item = Result<ScanMetadata, DeltaTableError>> + Send>>;

pub(super) async fn execution_plan(
    config: &DeltaScanConfig,
    session: &dyn Session,
    scan_plan: KernelScanPlan,
    stream: ScanMetadataStream,
    engine: Arc<dyn Engine>,
    limit: Option<usize>,
    file_selection: Option<&FileSelection>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let (files, transforms, dvs, metrics) = replay_files(
        session,
        engine,
        &scan_plan,
        config.clone(),
        stream,
        file_selection,
    )
    .await?;

    let file_id_field = scan_plan.contract.file_id_field.clone();
    if scan_plan.is_metadata_only() && !scan_plan.contract.retain_row_index {
        let map_file = |f: &ScanFileContext| {
            Ok((
                f.file_url.to_string(),
                match &f.stats.num_rows {
                    Precision::Exact(n) => *n,
                    _ => {
                        return plan_err!(
                            "Expected exact row counts in file: {}",
                            super::redact_url_for_error(&f.file_url)
                        );
                    }
                },
            ))
        };

        let maybe_file_rows = files
            .iter()
            .map(map_file)
            .try_collect::<_, VecDeque<_>, _>();
        if let Ok(file_rows) = maybe_file_rows {
            let retain_file_id = scan_plan.contract.retain_file_id;
            let exec = DeltaScanMetaExec::new(
                Arc::new(scan_plan),
                vec![file_rows],
                Arc::new(transforms),
                Arc::new(dvs),
                retain_file_id.then_some(file_id_field),
                metrics,
            );
            return Ok(Arc::new(exec) as _);
        }
    }

    get_data_scan_plan(session, scan_plan, files, transforms, dvs, metrics, limit).await
}

/// Materialize deletion vector keep masks for every file in the scan that has one.
///
/// Replay discovers the vectors and the loader fetches them, both running here
/// at once so a vector is on its way as soon as its file is seen. We drain the
/// full replay stream (discarding file contexts, stats, and partition values)
/// because discovery is a side effect of polling it.
pub(super) async fn replay_deletion_vectors(
    session: &dyn Session,
    engine: Arc<dyn Engine>,
    scan_plan: &KernelScanPlan,
    config: &DeltaScanConfig,
    stream: ScanMetadataStream,
) -> Result<Vec<DeletionVectorSelection>> {
    let table_root = scan_plan.scan.table_root().clone();
    let (dv_tx, dv_rx) = mpsc::unbounded_channel();
    let mut stream = ScanFileStream::new(&scan_plan.scan, config.clone(), None, stream, dv_tx);

    let replay = async {
        while stream.try_next().await?.is_some() {}
        stream.close_dv_input();
        Ok::<_, DeltaTableError>(())
    };
    let load = load_deletion_vectors(Arc::clone(session.runtime_env()), engine, table_root, dv_rx);
    let ((), loaded) = try_join!(replay, load)?;

    // Only files with a deletion vector are queued, so every result should
    // carry one. Guard with a typed error in case that invariant drifts.
    let dvs: DashMap<_, _> = loaded
        .into_iter()
        .map(|(url, dv, num_records)| match dv {
            Some(keep_mask) => normalize_dv_keep_mask_for_api(keep_mask, num_records, &url)
                .map(|mask| (url.to_string(), mask))
                .map_err(DeltaTableError::from),
            None => Err(DeltaTableError::generic(
                "Invariant violation: deletion vector queued for a file without one",
            )),
        })
        .collect::<std::result::Result<_, DeltaTableError>>()?;

    let mut vectors: Vec<_> = dvs
        .into_iter()
        .map(|(filepath, keep_mask)| DeletionVectorSelection {
            filepath,
            keep_mask,
        })
        .collect();
    vectors.sort_unstable_by(|left, right| left.filepath.cmp(&right.filepath));
    Ok(vectors)
}

async fn replay_files(
    session: &dyn Session,
    engine: Arc<dyn Engine>,
    scan_plan: &KernelScanPlan,
    scan_config: DeltaScanConfig,
    stream: ScanMetadataStream,
    file_selection: Option<&FileSelection>,
) -> Result<(
    Vec<ScanFileContext>,
    HashMap<String, Arc<Expression>>,
    DashMap<String, Vec<bool>>,
    ExecutionPlanMetricsSet,
)> {
    let table_root = scan_plan.scan.table_root().clone();
    let (dv_tx, dv_rx) = mpsc::unbounded_channel();
    let mut stream = ScanFileStream::new(
        &scan_plan.scan,
        scan_config,
        file_selection.map(|selection| &selection.file_ids),
        stream,
        dv_tx,
    );

    // Replay the file list and fetch deletion vectors at the same time: a
    // vector starts loading as soon as replay reaches its file.
    let (mut files, loaded_dvs) = try_join!(
        async {
            let mut files = Vec::new();
            while let Some(file) = stream.try_next().await? {
                files.extend(file);
            }
            stream.close_dv_input();
            Ok::<_, DeltaTableError>(files)
        },
        load_deletion_vectors(Arc::clone(session.runtime_env()), engine, table_root, dv_rx)
    )?;
    let dvs = into_keep_masks(loaded_dvs);

    if let Some(selection) = file_selection
        && selection.missing_file_policy == super::MissingFilePolicy::Error
    {
        let found: std::collections::HashSet<_> =
            files.iter().map(|f| f.file_url.to_string()).collect();
        let all_missing: Vec<_> = selection.file_ids.difference(&found).sorted().collect();

        if !all_missing.is_empty() {
            let missing_total = all_missing.len();
            let missing: Vec<_> = all_missing
                .iter()
                .take(10)
                .map(|id| super::redact_url_str_for_error(id))
                .collect();
            let extra = if missing_total > missing.len() {
                format!(" (and {} more)", missing_total - missing.len())
            } else {
                String::new()
            };
            return plan_err!(
                "File selection contains {missing_total} missing files (showing up to 10, redacted): {}{extra}",
                missing.join(", ")
            );
        }
    }

    let transforms: HashMap<_, _> = files
        .iter_mut()
        .flat_map(|file| {
            file.transform
                .take()
                .map(|t| (file.file_url.to_string(), t))
        })
        .collect();

    let metrics = ExecutionPlanMetricsSet::new();
    MetricBuilder::new(&metrics)
        .global_counter("count_files_scanned")
        .add(stream.metrics.num_scanned);

    Ok((files, transforms, dvs, metrics))
}

/// Normalize a DV keep mask for `deletion_vectors()`.
///
/// Kernel returns a sparse mask (up to the highest deleted row index). For API output we need one
/// full mask per file, to do this we pad trailing entries with `true` up to `numRecords`. If `numRecords`
/// is missing we fail, because we cannot know the correct full length.
///
/// This is API only. Scan execution does per batch normalization in `exec::consume_dv_mask` and
/// `exec_meta::apply_selection_vector`.
fn normalize_dv_keep_mask_for_api(
    mut mask: Vec<bool>,
    num_records: Option<u64>,
    file_url: &Url,
) -> Result<Vec<bool>> {
    let redacted_url = super::redact_url_for_error(file_url);
    let Some(num_records) = num_records else {
        return plan_err!(
            "Missing numRecords for file with deletion vector: {}",
            redacted_url
        );
    };
    let num_records = usize::try_from(num_records).map_err(|_| {
        DataFusionError::Execution(format!(
            "numRecords does not fit usize for file with deletion vector: {redacted_url}"
        ))
    })?;
    if mask.len() > num_records {
        return plan_err!(
            "Deletion vector mask length {} exceeds numRecords {} for file: {}",
            mask.len(),
            num_records,
            redacted_url
        );
    }
    mask.resize(num_records, true);
    Ok(mask)
}

async fn get_data_scan_plan(
    session: &dyn Session,
    scan_plan: KernelScanPlan,
    files: Vec<ScanFileContext>,
    transforms: HashMap<String, Arc<Expression>>,
    dvs: DashMap<String, Vec<bool>>,
    metrics: ExecutionPlanMetricsSet,
    limit: Option<usize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut partition_stats = HashMap::new();

    // Convert the files into datafusions `PartitionedFile`s grouped by the object store they are stored in
    // this is used to create a DataSourceExec plan for each store
    // To correlate the data with the original file, we add the file url as a partition value
    // This is required to apply the correct transform to the data in downstream processing.
    let to_partitioned_file = |f: ScanFileContext| {
        if let Some(part_stata) = &f.partitions {
            update_partition_stats(part_stata, &f.stats, &mut partition_stats)?;
        }
        // We create a PartitionedFile from the ObjectMeta to avoid any surprises in path encoding
        // that may arise from using the 'new' method directly. i.e. the 'new' method encodes paths
        // segments again, which may lead to double-encoding in some cases.
        let mut partitioned_file: PartitionedFile = ObjectMeta {
            location: Path::from_url_path(f.file_url.path())?,
            size: f.size,
            last_modified: Utc.timestamp_nanos(0),
            e_tag: None,
            version: None,
        }
        .into();
        let file_value = wrap_file_id_value(f.file_url.as_str());
        // NOTE: `PartitionedFile::with_statistics` appends exact stats for partition columns based
        // on `partition_values`, so partition values must be set first.
        partitioned_file.partition_values = vec![file_value.clone()];
        partitioned_file = partitioned_file.with_statistics(Arc::new(f.stats));
        Ok::<_, DataFusionError>((
            f.file_url.as_object_store_url(),
            (partitioned_file, None::<Vec<bool>>),
        ))
    };

    // Group the files by their object store url. Since datafusion assumes that all files in a
    // DataSourceExec are stored in the same object store, we need to create one plan per store
    let files_by_store = files
        .into_iter()
        .map(to_partitioned_file)
        .try_collect::<_, Vec<_>, _>()?
        .into_iter()
        .into_group_map();

    // TODO(roeap); not sure exactly how row tracking is implemented in kernel right now
    // so leaving predicate as None for now until we are sure this is safe to do.
    let table_config = scan_plan.table_configuration();
    let predicate = if table_config.is_feature_enabled(&TableFeature::RowTracking) {
        None
    } else {
        scan_plan.parquet_predicate.as_ref()
    };
    let file_id_field = scan_plan.contract.file_id_field.clone();
    let pq_plan = get_read_plan(
        session,
        files_by_store,
        &scan_plan.parquet_read_schema,
        &scan_plan.parquet_predicate_schema,
        limit,
        &file_id_field,
        predicate,
    )
    .await?;

    let exec = DeltaScanExec::new(
        Arc::new(scan_plan),
        pq_plan,
        Arc::new(transforms),
        Arc::new(dvs),
        partition_stats,
        metrics,
    );

    Ok(Arc::new(exec))
}

fn update_partition_stats(
    data: &StructData,
    stats: &Statistics,
    part_stats: &mut HashMap<String, ColumnStatistics>,
) -> Result<()> {
    for (field, stat) in data.fields().iter().zip(data.values().iter()) {
        let (null_count, value) = if stat.is_null() {
            (stats.num_rows, Precision::Absent)
        } else {
            (
                Precision::Exact(0),
                Precision::Exact(to_datafusion_scalar(stat)?),
            )
        };
        if let Some(part_stat) = part_stats.get_mut(field.name()) {
            part_stat.null_count = part_stat.null_count.add(&null_count);
            part_stat.min_value = part_stat.min_value.min(&value);
            part_stat.max_value = part_stat.max_value.max(&value);
        } else {
            part_stats.insert(
                field.name().clone(),
                ColumnStatistics {
                    null_count,
                    min_value: value.clone(),
                    max_value: value,
                    distinct_count: Precision::Absent,
                    sum_value: Precision::Absent,
                    byte_size: Precision::Absent,
                },
            );
        }
    }

    Ok(())
}

type FilesByStore = (ObjectStoreUrl, Vec<(PartitionedFile, Option<Vec<bool>>)>);

/// Maximum number of distinct values representable by DataFusion's default partition dictionary
/// encoding (`Dictionary<UInt16, _>`).
const MAX_PARTITION_DICT_CARDINALITY: usize = (u16::MAX as usize) + 1;

fn partitioned_files_to_file_groups(
    files: impl IntoIterator<Item = PartitionedFile>,
) -> Vec<FileGroup> {
    partitioned_files_to_file_groups_with_limit(files, MAX_PARTITION_DICT_CARDINALITY)
}

fn partitioned_files_to_file_groups_with_limit(
    files: impl IntoIterator<Item = PartitionedFile>,
    max_files_per_group: usize,
) -> Vec<FileGroup> {
    let file_groups = files
        .into_iter()
        // Each `PartitionedFile` is assigned to exactly one file group. DeltaScanStream stores
        // row ordinal counters per execution partition. Whole file ownership is required for
        // scan row ordinals.
        // Partition values are dictionary encoded using a UInt16 key (DataFusion's default
        // `wrap_partition_type_in_dict`). Keep file groups small enough that the file-id partition
        // dictionary doesn't exceed the key space (one distinct value per file).
        .chunks(max_files_per_group)
        .into_iter()
        .map(|chunk| chunk.collect::<FileGroup>())
        .collect_vec();

    #[cfg(debug_assertions)]
    {
        let mut owner_by_path = HashMap::new();
        for (partition, group) in file_groups.iter().enumerate() {
            for file in group.iter() {
                let path = file.object_meta.location.to_string();
                if let Some(previous_partition) = owner_by_path.insert(path.clone(), partition) {
                    debug_assert_eq!(
                        previous_partition, partition,
                        "file {path} was assigned to multiple scan partitions; row indexes require whole file ownership"
                    );
                }
            }
        }
    }

    file_groups
}

/// A [`PhysicalExprAdapterFactory`] that resolves column-mapped reads by Parquet
/// field id instead of physical name.
///
/// Under `columnMapping.mode=id` the read schema names columns by Delta physical name
/// (`col-<id>`) and stamps `PARQUET:field_id`, but a native Iceberg writer (e.g. a
/// Unity Catalog Uniform table over Flink/pyiceberg) names them logically (`op`,
/// `after`) under the same field id. [`DefaultPhysicalExprAdapter`] matches by name
/// only, so `col-<id>` is never found on disk: a non-nullable column errors, a
/// nullable one reads NULL, and a struct fails the by-name cast.
///
/// This bridges the two by field id at every level: it relabels the read schema to
/// the file's names so the default adapter resolves and leaf-coerces by name, then
/// rebuilds columns whose nested names diverge back to `col-<id>` via
/// [`FieldIdRealignExpr`]. Native Delta or `name` mode: field ids match, so it is a
/// no-op that behaves like the default.
#[derive(Debug)]
struct FieldIdAlignedExprAdapterFactory {
    inner: DefaultPhysicalExprAdapterFactory,
}

impl Default for FieldIdAlignedExprAdapterFactory {
    fn default() -> Self {
        Self {
            inner: DefaultPhysicalExprAdapterFactory {},
        }
    }
}

impl PhysicalExprAdapterFactory for FieldIdAlignedExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Result<Arc<dyn PhysicalExprAdapter>> {
        let alignment = align_read_schema_to_file_names(&logical_file_schema, &physical_file_schema);
        // The default adapter resolves and leaf-coerces by name, so give it the
        // file-aligned read schema. It yields file-named arrays; `realign_targets`
        // rebuilds those with diverging nested names back to `col-<id>` (see below).
        let inner = self
            .inner
            .create(alignment.file_aligned_read_schema, physical_file_schema)?;
        Ok(Arc::new(FieldIdAlignedExprAdapter {
            inner,
            read_to_file: alignment.read_to_file,
            realign_targets: alignment.realign_targets,
        }))
    }
}

#[derive(Debug)]
struct FieldIdAlignedExprAdapter {
    inner: Arc<dyn PhysicalExprAdapter>,
    /// Read physical name (`col-<id>`) -> file name, for fields sharing a
    /// `PARQUET:field_id` but named differently. Empty means pass-through to the default.
    read_to_file: HashMap<String, String>,
    /// Read physical name (`col-<id>`) -> read field, for columns whose nested
    /// struct/list/map child names diverge from the file. A by-name cast cannot rename
    /// children, so [`FieldIdRealignExpr`] rebuilds them.
    realign_targets: HashMap<String, FieldRef>,
}

impl PhysicalExprAdapter for FieldIdAlignedExprAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        if self.read_to_file.is_empty() && self.realign_targets.is_empty() {
            return self.inner.rewrite(expr);
        }
        // Rename each column from `col-<id>` to the file name sharing its field id so
        // the default adapter resolves it by name. Columns with diverging nested names
        // are also wrapped in `FieldIdRealignExpr`; the inner column is still resolved
        // and leaf-coerced by the default adapter's pass below.
        let aligned = expr
            .transform_down(|node| match node.as_any().downcast_ref::<Column>() {
                Some(column) => {
                    let file_name = self
                        .read_to_file
                        .get(column.name())
                        .map(String::as_str)
                        .unwrap_or_else(|| column.name());
                    let file_column: Arc<dyn PhysicalExpr> =
                        Arc::new(Column::new(file_name, column.index()));
                    match self.realign_targets.get(column.name()) {
                        Some(target) => Ok(Transformed::yes(Arc::new(FieldIdRealignExpr::new(
                            file_column,
                            target.data_type().clone(),
                        )))),
                        None if file_name != column.name() => Ok(Transformed::yes(file_column)),
                        None => Ok(Transformed::no(node)),
                    }
                }
                None => Ok(Transformed::no(node)),
            })?
            .data;
        self.inner.rewrite(aligned)
    }
}

/// Field-id alignment between the read schema (`col-<id>` names) and the data file
/// (an Iceberg writer's logical names).
struct ReadSchemaAlignment {
    /// Read schema relabeled to the file's names at every level; the inner adapter's
    /// read schema, so it resolves and leaf-coerces by name.
    file_aligned_read_schema: SchemaRef,
    /// Top-level `col-<id>` -> file name, for rewriting column expressions.
    read_to_file: HashMap<String, String>,
    /// Top-level `col-<id>` -> read field, for columns whose nested names diverge and
    /// must be rebuilt by field id.
    realign_targets: HashMap<String, FieldRef>,
}

/// Aligns the read schema to the data file by `PARQUET:field_id`.
///
/// Relabeling recurses into struct/list/map so the file-aligned schema carries the
/// file's names at every level, letting the inner adapter resolve and leaf-cast
/// against logical-named Iceberg data. Only names change; types and nullability are
/// kept. Fields with no field id, or an id absent from the file, are left as is.
///
/// `read_to_file` holds top-level renames (column exprs reference top-level names).
/// `realign_targets` holds columns whose nested names diverge, rebuilt by field id
/// since a by-name struct cast cannot rename children. Empty maps: nothing to bridge.
fn align_read_schema_to_file_names(read: &Schema, file: &Schema) -> ReadSchemaAlignment {
    let file_field_by_id = fields_by_field_id(file.fields());
    let mut read_to_file = HashMap::new();
    let mut realign_targets = HashMap::new();
    let fields: Vec<FieldRef> = read
        .fields()
        .iter()
        .map(|read_field| {
            match field_id(read_field).and_then(|id| file_field_by_id.get(id)) {
                Some(file_field) => {
                    if file_field.name() != read_field.name() {
                        read_to_file
                            .insert(read_field.name().clone(), file_field.name().to_string());
                    }
                    let aligned = align_field_to_file(read_field, file_field);
                    // Type changes only when a nested name diverges (a top-level rename
                    // rides on the column expression, not the type). Record those to rebuild.
                    if aligned.data_type() != read_field.data_type() {
                        realign_targets.insert(read_field.name().clone(), read_field.clone());
                    }
                    Arc::new(aligned)
                }
                None => read_field.clone(),
            }
        })
        .collect();
    let file_aligned_read_schema =
        Arc::new(Schema::new(fields).with_metadata(read.metadata().clone()));
    ReadSchemaAlignment {
        file_aligned_read_schema,
        read_to_file,
        realign_targets,
    }
}

/// Index a field list by `PARQUET:field_id`, skipping fields without one.
fn fields_by_field_id(fields: &arrow_schema::Fields) -> HashMap<&str, &FieldRef> {
    fields
        .iter()
        .filter_map(|f| field_id(f).map(|id| (id, f)))
        .collect()
}

/// The read field renamed to the file field's name, its nested children recursively
/// realigned to the file's names. Types and nullability are preserved.
fn align_field_to_file(read: &FieldRef, file: &FieldRef) -> Field {
    read.as_ref()
        .clone()
        .with_name(file.name())
        .with_data_type(align_type_to_file_names(read.data_type(), file.data_type()))
}

/// Recursively relabel the read type's nested field names to the file's names where
/// they share a `PARQUET:field_id`. Only struct/list/map recurse; leaves and any id
/// absent from the file are returned unchanged. Shape and leaf types are preserved.
fn align_type_to_file_names(read: &DataType, file: &DataType) -> DataType {
    use DataType::{LargeList, List, Map, Struct};
    match (read, file) {
        (Struct(read_children), Struct(file_children)) => {
            let file_field_by_id = fields_by_field_id(file_children);
            let aligned: Vec<FieldRef> = read_children
                .iter()
                .map(
                    |child| match field_id(child).and_then(|id| file_field_by_id.get(id)) {
                        Some(file_child) => Arc::new(align_field_to_file(child, file_child)),
                        None => child.clone(),
                    },
                )
                .collect();
            Struct(aligned.into())
        }
        (List(read_inner), List(file_inner)) => List(Arc::new(align_field_to_file(
            read_inner, file_inner,
        ))),
        (LargeList(read_inner), LargeList(file_inner)) => {
            LargeList(Arc::new(align_field_to_file(read_inner, file_inner)))
        }
        (Map(read_entries, sorted), Map(file_entries, _)) => {
            Map(Arc::new(align_field_to_file(read_entries, file_entries)), *sorted)
        }
        _ => read.clone(),
    }
}

fn field_id(field: &Field) -> Option<&str> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .map(String::as_str)
}

/// A [`PhysicalExpr`] that rebuilds its input array to `target` by Parquet field id,
/// renaming nested struct/list/map children to the target's names.
///
/// The default adapter resolves a column by name, yielding (for an Iceberg file) a
/// struct with the file's logical child names. A by-name cast cannot rename children,
/// so this expr pairs each target child with the source child sharing its field id
/// (falling back to position) and recurses, reusing leaf arrays. It runs before the
/// kernel transform, which validates its input against the physical (`col-<id>`) schema.
#[derive(Debug, Clone, Eq)]
struct FieldIdRealignExpr {
    input: Arc<dyn PhysicalExpr>,
    target: DataType,
}

impl FieldIdRealignExpr {
    fn new(input: Arc<dyn PhysicalExpr>, target: DataType) -> Self {
        Self { input, target }
    }
}

// Manually implement PartialEq and Hash, mirroring DataFusion's own physical
// expressions (a derive on `Arc<dyn PhysicalExpr>` hits rust-lang/rust#78808).
impl PartialEq for FieldIdRealignExpr {
    fn eq(&self, other: &Self) -> bool {
        self.input.eq(&other.input) && self.target.eq(&other.target)
    }
}

impl std::hash::Hash for FieldIdRealignExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.target.hash(state);
    }
}

impl std::fmt::Display for FieldIdRealignExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FieldIdRealign({}, {})", self.input, self.target)
    }
}

impl PhysicalExpr for FieldIdRealignExpr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.target.clone())
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.input.nullable(input_schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let array = self.input.evaluate(batch)?.into_array(batch.num_rows())?;
        Ok(ColumnarValue::Array(realign_array(&array, &self.target)?))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        match <[_; 1]>::try_from(children) {
            Ok([input]) => Ok(Arc::new(Self::new(input, self.target.clone()))),
            Err(children) => plan_err!(
                "FieldIdRealignExpr expects exactly one child, got {}",
                children.len()
            ),
        }
    }

    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// Rebuild `array` to `target`, renaming nested struct/list/map children by field id
/// and casting leaves whose type differs. Leaf arrays are reused, so this is cheap.
fn realign_array(array: &ArrayRef, target: &DataType) -> Result<ArrayRef> {
    use arrow_array::{Array, LargeListArray, ListArray, MapArray, StructArray};
    match target {
        DataType::Struct(target_fields) => {
            let source = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| realign_type_error("struct", array))?;
            let source_idx_by_id = field_index_by_field_id(source.fields());
            let children = target_fields
                .iter()
                .enumerate()
                .map(|(position, target_field)| {
                    let source_index = field_id(target_field)
                        .and_then(|id| source_idx_by_id.get(id).copied())
                        .unwrap_or(position);
                    let source_child = source.columns().get(source_index).ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "field-id realign found no source child for '{}'",
                            target_field.name()
                        ))
                    })?;
                    realign_array(source_child, target_field.data_type())
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(StructArray::try_new(
                target_fields.clone(),
                children,
                source.nulls().cloned(),
            )?))
        }
        DataType::List(target_inner) => {
            let source = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| realign_type_error("list", array))?;
            let values = realign_array(source.values(), target_inner.data_type())?;
            Ok(Arc::new(ListArray::try_new(
                target_inner.clone(),
                source.offsets().clone(),
                values,
                source.nulls().cloned(),
            )?))
        }
        DataType::LargeList(target_inner) => {
            let source = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| realign_type_error("large list", array))?;
            let values = realign_array(source.values(), target_inner.data_type())?;
            Ok(Arc::new(LargeListArray::try_new(
                target_inner.clone(),
                source.offsets().clone(),
                values,
                source.nulls().cloned(),
            )?))
        }
        DataType::Map(target_entries, sorted) => {
            let source = array
                .as_any()
                .downcast_ref::<MapArray>()
                .ok_or_else(|| realign_type_error("map", array))?;
            let entries = realign_array(
                &(Arc::new(source.entries().clone()) as ArrayRef),
                target_entries.data_type(),
            )?;
            let entries = entries
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("realign of map entries yields a struct array")
                .clone();
            Ok(Arc::new(MapArray::try_new(
                target_entries.clone(),
                source.offsets().clone(),
                entries,
                source.nulls().cloned(),
                *sorted,
            )?))
        }
        _ if array.data_type() == target => Ok(Arc::clone(array)),
        _ => Ok(cast_with_options(
            array.as_ref(),
            target,
            &CastOptions::default(),
        )?),
    }
}

fn realign_type_error(expected: &str, array: &ArrayRef) -> DataFusionError {
    DataFusionError::Internal(format!(
        "field-id realign expected a {expected} array, got {}",
        array.data_type()
    ))
}

/// Map `PARQUET:field_id` -> child index for a field list, skipping fields without one.
fn field_index_by_field_id(fields: &arrow_schema::Fields) -> HashMap<&str, usize> {
    fields
        .iter()
        .enumerate()
        .filter_map(|(index, f)| field_id(f).map(|id| (id, index)))
        .collect()
}

async fn get_read_plan(
    state: &dyn Session,
    files_by_store: impl IntoIterator<Item = FilesByStore>,
    // Schema of physical file columns to read from Parquet (no Delta partitions, no file-id).
    //
    // This is also the schema used for Parquet pruning/pushdown. It may include view types
    // (e.g. Utf8View/BinaryView) depending on `DeltaScanConfig`.
    parquet_read_schema: &SchemaRef,
    // Predicate binding schema used to bind Parquet predicates, including the synthetic file id
    // column when the provider exposes it.
    parquet_predicate_schema: &SchemaRef,
    limit: Option<usize>,
    file_id_field: &FieldRef,
    predicate: Option<&Expr>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut plans = Vec::new();

    let pq_options = TableParquetOptions {
        global: state.config().options().execution.parquet.clone(),
        ..Default::default()
    };

    let mut full_read_schema = SchemaBuilder::from(parquet_read_schema.as_ref().clone());
    full_read_schema.push(file_id_field.as_ref().clone().with_nullable(true));
    let full_read_schema = Arc::new(full_read_schema.finish());
    let parquet_predicate_df_schema = parquet_predicate_schema.clone().to_dfschema()?;
    let adapter_factory = Arc::new(FieldIdAlignedExprAdapterFactory::default());

    for (store_url, files) in files_by_store.into_iter() {
        let reader_factory = Arc::new(CachedParquetFileReaderFactory::new(
            state.runtime_env().object_store(&store_url)?,
            state.runtime_env().cache_manager.get_file_metadata_cache(),
        ));

        // NOTE: In the "next" provider, DataFusion's Parquet scan partition fields are file-id
        // only. Delta partition columns/values are injected via kernel transforms and handled
        // above Parquet, so they are not part of the Parquet partition schema here.
        let table_schema =
            TableSchema::new(parquet_read_schema.clone(), vec![file_id_field.clone()]);
        let full_table_schema = table_schema.table_schema().clone();
        let mut file_source = ParquetSource::new(table_schema)
            .with_table_parquet_options(pq_options.clone())
            .with_parquet_file_reader_factory(reader_factory);

        // TODO(roeap); we might be able to also push selection vectors into the read plan
        // by creating parquet access plans. However we need to make sure this does not
        // interfere with other delta features like row ids.
        let has_selection_vectors = files.iter().any(|(_, sv)| sv.is_some());
        if !has_selection_vectors && let Some(pred) = predicate {
            match state.create_physical_expr(pred.clone(), &parquet_predicate_df_schema) {
                Ok(physical) => match adapter_factory
                    .create(parquet_predicate_schema.clone(), full_read_schema.clone())
                {
                    Ok(adapter) => match adapter.rewrite(physical) {
                        Ok(rewritten) => {
                            file_source = file_source
                                .with_predicate(rewritten)
                                .with_pushdown_filters(true);
                        }
                        Err(err) => {
                            debug!(
                                predicate = ?pred,
                                schema = ?parquet_predicate_schema,
                                error = %err,
                                "Skipping parquet predicate pushdown because predicate adaptation to the read schema failed"
                            );
                        }
                    },
                    Err(err) => {
                        debug!(
                            predicate = ?pred,
                            schema = ?parquet_predicate_schema,
                            error = %err,
                            "Skipping parquet predicate pushdown because predicate adapter creation failed"
                        );
                    }
                },
                Err(err) => {
                    debug!(
                        predicate = ?pred,
                        schema = ?parquet_predicate_schema,
                        error = %err,
                        "Skipping parquet predicate pushdown because predicate binding failed"
                    );
                }
            }
        }

        let file_groups = partitioned_files_to_file_groups(files.into_iter().map(|file| file.0));
        let (file_groups, statistics) =
            compute_all_files_statistics(file_groups, full_table_schema, true, false)?;

        let config = FileScanConfigBuilder::new(store_url, Arc::new(file_source))
            .with_file_groups(file_groups)
            .with_statistics(statistics)
            .with_limit(limit)
            .with_expr_adapter(Some(adapter_factory.clone() as _))
            .build();

        plans.push(DataSourceExec::from_data_source(config) as Arc<dyn ExecutionPlan>);
    }

    Ok(match plans.len() {
        0 => Arc::new(EmptyExec::new(full_read_schema.clone())),
        1 => plans.remove(0),
        _ => UnionExec::try_new(plans)?,
    })
}

// Small helper to reuse some code between exec and exec_meta
fn finalize_transformed_batch(
    batch: RecordBatch,
    scan_plan: &KernelScanPlan,
    file_id_col: Option<(ArrayRef, FieldRef)>,
    schema_adapter: &mut SchemaAdapter,
) -> Result<RecordBatch> {
    let result = if let Some(projection) = scan_plan.contract.result_projection.as_ref() {
        batch.project(projection)?
    } else {
        batch
    };
    // NOTE: most data is read properly typed already, however columns added via
    // literals in the transformations may need to be cast to the physical expected type.
    let result = if result.schema_ref().eq(&scan_plan.contract.result_schema) {
        result
    } else {
        schema_adapter.adapt(result)?
    };
    if let Some((arr, field)) = file_id_col {
        let arr = if arr.data_type() != field.data_type() {
            let options = CastOptions {
                safe: true,
                ..Default::default()
            };
            cast_with_options(arr.as_ref(), field.data_type(), &options)?
        } else {
            arr
        };
        let mut columns = result.columns().to_vec();
        columns.push(arr);
        let mut fields = result.schema().fields().to_vec();
        fields.push(field);
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    } else {
        Ok(result)
    }
}

/// Caches a [`BatchAdapter`] for the most recently seen source schema, avoiding
/// repeated expression-tree construction when consecutive batches share the same
/// physical schema (the common case within a single file).
struct SchemaAdapter {
    factory: BatchAdapterFactory,
    /// Single-entry cache: the source schema for the currently cached adapter.
    cached_source: Option<SchemaRef>,
    cached_adapter: Option<BatchAdapter>,
}

impl SchemaAdapter {
    fn new(target_schema: SchemaRef) -> Self {
        Self {
            factory: BatchAdapterFactory::new(target_schema),
            cached_source: None,
            cached_adapter: None,
        }
    }

    /// Adapt the batch to the target schema, using a cached adapter when the
    /// source schema matches the previous call.
    fn adapt(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        let source_schema = batch.schema();
        let can_reuse = matches!(
            (&self.cached_source, &self.cached_adapter),
            (Some(cached_source), Some(_)) if cached_source.eq(&source_schema)
        );
        let needs_rebuild = !can_reuse;
        if needs_rebuild {
            let adapter = self.factory.make_adapter(&source_schema)?;
            self.cached_source = Some(source_schema);
            self.cached_adapter = Some(adapter);
        }
        match self.cached_adapter.as_ref() {
            Some(adapter) => adapter.adapt_batch(&batch),
            None => plan_err!(
                "schema adapter cache entry missing for source schema: {:?}",
                batch.schema()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::{
        BinaryArray, BinaryViewArray, Int32Array, Int64Array, RecordBatch, RecordBatchOptions,
        StringArray, StringViewArray, StructArray,
    };
    use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
    use datafusion::{
        error::DataFusionError,
        physical_plan::collect,
        prelude::{col, lit},
    };
    use object_store::{ObjectStoreExt as _, memory::InMemory};
    use parquet::arrow::ArrowWriter;
    use url::Url;

    use crate::{
        assert_batches_sorted_eq,
        delta_datafusion::{session::create_session, table_provider::next::FILE_ID_COLUMN_DEFAULT},
        test_utils::TestResult,
    };

    use super::{plan::build_parquet_predicate_schema, *};

    /// Build a `Utf8` field carrying a `PARQUET:field_id`, mirroring how the
    /// kernel/Parquet reader stamps column-mapping ids onto Arrow schemas.
    fn field_with_id(name: &str, field_id: &str, nullable: bool) -> Field {
        Field::new(name, DataType::Utf8, nullable)
            .with_metadata([(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string())].into())
    }

    #[test]
    fn test_align_read_schema_pairs_top_level_fields_on_field_id() {
        // Read schema uses Delta physical names (col-<id>); the file uses logical
        // names. The struct child must keep its read name (col-104) -- it is the
        // cast target -- while id 9, absent from the file, is left as col-9.
        let read = Schema::new(vec![
            field_with_id("col-102", "102", false),
            Field::new(
                "col-103",
                DataType::Struct(vec![field_with_id("col-104", "104", true)].into()),
                true,
            )
            .with_metadata([(PARQUET_FIELD_ID_META_KEY.to_string(), "103".to_string())].into()),
            field_with_id("col-9", "9", true),
        ]);
        let file = Schema::new(vec![
            field_with_id("op", "102", false),
            Field::new(
                "after",
                DataType::Struct(vec![field_with_id("amount", "104", true)].into()),
                true,
            )
            .with_metadata([(PARQUET_FIELD_ID_META_KEY.to_string(), "103".to_string())].into()),
        ]);

        let alignment = align_read_schema_to_file_names(&read, &file);
        let relabeled = &alignment.file_aligned_read_schema;
        let map = &alignment.read_to_file;

        // Top-level fields take the file names; read-side nullability is preserved.
        assert_eq!(relabeled.field(0).name(), "op");
        assert!(!relabeled.field(0).is_nullable());
        let after = relabeled.field(1);
        assert_eq!(after.name(), "after");
        // The struct child is relabeled to the file name by field id, so the inner
        // adapter's name-based cast succeeds against the logically-named file.
        match after.data_type() {
            DataType::Struct(children) => assert_eq!(children[0].name(), "amount"),
            other => panic!("expected struct, got {other:?}"),
        }
        // The unmatched field is untouched.
        assert_eq!(relabeled.field(2).name(), "col-9");

        assert_eq!(map.get("col-102").map(String::as_str), Some("op"));
        assert_eq!(map.get("col-103").map(String::as_str), Some("after"));
        assert!(!map.contains_key("col-9"));

        // The struct column's nested name diverges, so it is recorded for field-id
        // reconstruction; the scalar `op` is not (a column expression's top-level
        // rename suffices).
        assert!(alignment.realign_targets.contains_key("col-103"));
        assert!(!alignment.realign_targets.contains_key("col-102"));
    }

    #[test]
    fn test_align_read_schema_is_noop_without_field_ids() {
        // Native Delta (or `name` mode): no field ids, names already match. The
        // relabel must leave the schema untouched and the map empty so the default
        // adapter behaves exactly as before.
        let read = Schema::new(vec![Field::new("col-3877", DataType::Utf8, true)]);
        let file = Schema::new(vec![Field::new("col-3877", DataType::Utf8, true)]);

        let alignment = align_read_schema_to_file_names(&read, &file);

        assert_eq!(alignment.file_aligned_read_schema.as_ref(), &read);
        assert!(alignment.read_to_file.is_empty());
        assert!(alignment.realign_targets.is_empty());
    }

    #[test]
    fn test_realign_array_rebuilds_struct_children_by_field_id() {
        // Source struct mirrors a logically-named Iceberg file: children carry the
        // file's names plus `PARQUET:field_id`, and are listed in a different order
        // than the target to prove matching is by field id, not position.
        let source = StructArray::try_new(
            vec![
                Arc::new(field_with_id("merchant", "105", true)),
                Arc::new(field_with_id("id", "104", true)),
            ]
            .into(),
            vec![
                Arc::new(StringArray::from(vec![Some("Coffee Shop"), Some("Gas")])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("txn-1"), Some("txn-2")])) as ArrayRef,
            ],
            None,
        )
        .unwrap();

        // Target uses Delta physical names (`col-<id>`) and Utf8View, forcing both a
        // rename and a leaf cast.
        let field_id_meta = |id: &str| {
            [(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>()
        };
        let target = DataType::Struct(
            vec![
                Field::new("col-104", DataType::Utf8View, true).with_metadata(field_id_meta("104")),
                Field::new("col-105", DataType::Utf8View, true).with_metadata(field_id_meta("105")),
            ]
            .into(),
        );

        let out = realign_array(&(Arc::new(source) as ArrayRef), &target).unwrap();
        assert_eq!(out.data_type(), &target);

        let out = out.as_any().downcast_ref::<StructArray>().unwrap();
        let id = out
            .column_by_name("col-104")
            .unwrap()
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        assert_eq!(id.value(0), "txn-1");
        let merchant = out
            .column_by_name("col-105")
            .unwrap()
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        assert_eq!(merchant.value(0), "Coffee Shop");
    }

    #[test]
    fn test_realign_array_recurses_through_list_of_struct() {
        // Exercises the List branch: a `list<struct>` whose struct children carry the
        // file's logical names is realigned to `col-<id>` names, reusing the list
        // offsets/nulls and renaming the nested struct by field id.
        use arrow_array::ListArray;
        use arrow_buffer::OffsetBuffer;

        let field_id_meta = |id: &str| {
            [(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>()
        };

        // Two list rows: [ {id: a}, {id: b} ] and [ {id: c} ]. The struct child uses the
        // file's logical name `id` plus field id 104.
        let structs = StructArray::try_new(
            vec![Arc::new(field_with_id("id", "104", true))].into(),
            vec![Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                Some("c"),
            ])) as ArrayRef],
            None,
        )
        .unwrap();
        let source_inner =
            Field::new("item", structs.data_type().clone(), true).with_metadata(field_id_meta("103"));
        let source = ListArray::try_new(
            Arc::new(source_inner),
            OffsetBuffer::from_lengths([2, 1]),
            Arc::new(structs) as ArrayRef,
            None,
        )
        .unwrap();

        // Target list inner is a struct with the Delta physical name `col-104` and
        // Utf8View, forcing both a nested rename and a leaf cast.
        let target_inner = Field::new(
            "item",
            DataType::Struct(
                vec![Field::new("col-104", DataType::Utf8View, true)
                    .with_metadata(field_id_meta("104"))]
                .into(),
            ),
            true,
        )
        .with_metadata(field_id_meta("103"));
        let target = DataType::List(Arc::new(target_inner));

        let out = realign_array(&(Arc::new(source) as ArrayRef), &target).unwrap();
        assert_eq!(out.data_type(), &target);

        let out = out.as_any().downcast_ref::<ListArray>().unwrap();
        // Offsets are reused unchanged: row 0 spans two elements, row 1 spans one.
        assert_eq!(out.value_offsets(), &[0, 2, 3]);
        let items = out
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let ids = items
            .column_by_name("col-104")
            .expect("nested child renamed to its physical name")
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        assert_eq!(ids.value(0), "a");
        assert_eq!(ids.value(2), "c");
    }

    #[test]
    fn test_realign_array_leaf_is_identity_when_types_match() {
        let array = Arc::new(StringArray::from(vec![Some("a")])) as ArrayRef;
        let out = realign_array(&array, &DataType::Utf8).unwrap();
        assert_eq!(out.as_ref(), array.as_ref());
    }

    #[test]
    fn test_partitioned_files_to_file_groups_respects_dictionary_cardinality_limit() {
        let files = (0..=MAX_PARTITION_DICT_CARDINALITY)
            .map(|i| PartitionedFile::new(format!("memory:///f{i}.parquet"), 0))
            .collect_vec();

        let groups = partitioned_files_to_file_groups(files);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), MAX_PARTITION_DICT_CARDINALITY);
        assert_eq!(groups[1].len(), 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "row indexes require whole file ownership")]
    fn test_partitioned_files_to_file_groups_rejects_split_file_across_groups_in_debug() {
        let files = vec![
            PartitionedFile::new("memory:///same.parquet", 0),
            PartitionedFile::new("memory:///other.parquet", 0),
            PartitionedFile::new("memory:///same.parquet", 0),
        ];

        let _ = partitioned_files_to_file_groups_with_limit(files, 1);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_pads_short_mask_with_true() {
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let actual = normalize_dv_keep_mask_for_api(vec![true, false], Some(4), &url).unwrap();
        assert_eq!(actual, vec![true, false, true, true]);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_keeps_equal_length_mask() {
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let mask = vec![true, false, true];
        let actual = normalize_dv_keep_mask_for_api(mask.clone(), Some(3), &url).unwrap();
        assert_eq!(actual, mask);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_pads_empty_mask_to_all_true() {
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let actual = normalize_dv_keep_mask_for_api(Vec::new(), Some(3), &url).unwrap();
        assert_eq!(actual, vec![true, true, true]);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_errors_when_mask_longer_than_num_records() {
        let url =
            Url::parse("s3://user:secret@example.com/table/file.parquet?sig=token#frag").unwrap();
        let expected_url = super::super::redact_url_for_error(&url);
        let err = normalize_dv_keep_mask_for_api(vec![true, false, true], Some(2), &url)
            .expect_err("longer mask should error");
        let message = err.to_string();
        assert!(message.contains("exceeds numRecords"));
        assert!(message.contains(&expected_url));
        assert!(!message.contains("sig=token"));
        assert!(!message.contains("secret"));
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_errors_when_num_records_missing() {
        let url =
            Url::parse("s3://user:secret@example.com/table/file.parquet?sig=token#frag").unwrap();
        let expected_url = super::super::redact_url_for_error(&url);
        let err = normalize_dv_keep_mask_for_api(vec![true], None, &url)
            .expect_err("missing numRecords should error");
        let message = err.to_string();
        assert!(message.contains("Missing numRecords"));
        assert!(message.contains(&expected_url));
        assert!(!message.contains("sig=token"));
        assert!(!message.contains("secret"));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn test_normalize_dv_keep_mask_for_api_errors_when_num_records_overflow_usize() {
        // This branch is only reachable on 32-bit targets where u64 may exceed usize.
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let overflow_num_records = (usize::MAX as u64) + 1;
        let err = normalize_dv_keep_mask_for_api(vec![true], Some(overflow_num_records), &url)
            .expect_err("numRecords that does not fit usize should error");
        assert!(err.to_string().contains("does not fit usize"));
    }

    #[test]
    fn test_schema_adapter_synthesizes_nullable_columns() {
        let source_schema = Arc::new(Schema::new(Fields::empty()));
        let source = RecordBatch::try_new_with_options(
            source_schema,
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(2)),
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let mut adapter = SchemaAdapter::new(target_schema.clone());
        let adapted = adapter.adapt(source).unwrap();

        assert_eq!(adapted.schema().as_ref(), target_schema.as_ref());
        assert_eq!(adapted.num_rows(), 2);
        let id = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id.null_count(), 2);
    }

    #[test]
    fn test_schema_adapter_missing_non_nullable_column_errors() {
        let source_schema = Arc::new(Schema::new(Fields::empty()));
        let source = RecordBatch::try_new_with_options(
            source_schema,
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let mut adapter = SchemaAdapter::new(target_schema);
        let err = adapter
            .adapt(source)
            .expect_err("missing non-nullable columns should error");
        match err {
            DataFusionError::Execution(msg) => {
                assert!(
                    msg.contains("Non-nullable column 'id'"),
                    "expected non-nullable missing-column error, got: {msg}"
                );
                assert!(
                    msg.contains("missing from the physical schema"),
                    "expected missing physical schema detail, got: {msg}"
                );
            }
            other => {
                panic!("expected execution error for missing non-nullable column, got: {other}")
            }
        }
    }

    #[test]
    fn test_schema_adapter_invalid_scalar_cast_errors() {
        let source_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, true)]));
        let source = RecordBatch::try_new(
            source_schema,
            vec![Arc::new(StringArray::from(vec![Some("not-an-int")]))],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let mut adapter = SchemaAdapter::new(target_schema);
        let err = adapter
            .adapt(source)
            .expect_err("invalid value cast should fail under DataFusion default cast semantics");
        match err {
            DataFusionError::ArrowError(inner, _) => {
                assert!(
                    matches!(inner.as_ref(), ArrowError::CastError(_)),
                    "expected arrow cast error, got: {inner}"
                );
            }
            other => panic!("expected arrow cast error for invalid scalar cast, got: {other}"),
        }
    }

    #[test]
    fn test_schema_adapter_type_widening() {
        let source_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let source = RecordBatch::try_new(
            source_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            ],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let mut adapter = SchemaAdapter::new(target_schema.clone());
        let adapted = adapter.adapt(source).unwrap();

        assert_eq!(adapted.schema().as_ref(), target_schema.as_ref());
        assert_eq!(adapted.num_rows(), 3);
        let id = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id.values(), &[1i64, 2, 3]);
    }

    #[test]
    fn test_schema_adapter_overflow_cast_errors() {
        let source_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
        let source = RecordBatch::try_new(
            source_schema,
            vec![Arc::new(Int64Array::from(vec![i64::from(i32::MAX) + 1]))],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let mut adapter = SchemaAdapter::new(target_schema);
        let err = adapter
            .adapt(source)
            .expect_err("overflow cast should fail under DataFusion default cast semantics");
        match err {
            DataFusionError::ArrowError(inner, _) => {
                assert!(
                    matches!(inner.as_ref(), ArrowError::CastError(_)),
                    "expected arrow cast error, got: {inner}"
                );
            }
            other => panic!("expected arrow cast error for overflow cast, got: {other}"),
        }
    }

    #[test]
    fn test_schema_adapter_caches_across_calls() {
        let source_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let mut adapter = SchemaAdapter::new(target_schema);

        let batch1 = RecordBatch::try_new(
            Arc::clone(&source_schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            Arc::clone(&source_schema),
            vec![Arc::new(Int32Array::from(vec![2]))],
        )
        .unwrap();

        let _ = adapter.adapt(batch1).unwrap();
        assert!(adapter.cached_source.is_some());

        // Second call with the same schema should hit the cache (no rebuild).
        let adapted = adapter.adapt(batch2).unwrap();
        assert_eq!(adapted.num_rows(), 1);
        let id = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id.values(), &[2i64]);
    }

    #[tokio::test]
    async fn test_parquet_plan() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_data.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_data.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 1  | a     | memory:///test_data.parquet |",
            "| 2  | b     | memory:///test_data.parquet |",
            "| 3  | c     | memory:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        // respect limits
        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            Some(1),
            &file_id_field,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 1  | a     | memory:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        // extended schema with missing column
        let arrow_schema_extended = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
            Field::new("value2", DataType::Utf8, true),
        ]));
        let parquet_predicate_schema_extended =
            build_parquet_predicate_schema(&arrow_schema_extended, &file_id_field);
        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema_extended,
            &parquet_predicate_schema_extended,
            Some(1),
            &file_id_field,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+--------+-----------------------------+",
            "| id | value | value2 | __delta_rs_file_id__        |",
            "+----+-------+--------+-----------------------------+",
            "| 1  | a     |        | memory:///test_data.parquet |",
            "+----+-------+--------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_parquet_plan_nested() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let nested_fields: Fields = vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Utf8, true),
        ]
        .into();
        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("nested", DataType::Struct(nested_fields.clone()), true),
        ]));
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StructArray::try_new(
                    nested_fields,
                    vec![
                        Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
                        Arc::new(StringArray::from(vec![Some("aa"), Some("bb"), Some("cc")])),
                    ],
                    None,
                )?),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_data.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_data.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+---------------+-----------------------------+",
            "| id | nested        | __delta_rs_file_id__        |",
            "+----+---------------+-----------------------------+",
            "| 1  | {a: a, b: aa} | memory:///test_data.parquet |",
            "| 2  | {a: b, b: bb} | memory:///test_data.parquet |",
            "| 3  | {a: c, b: cc} | memory:///test_data.parquet |",
            "+----+---------------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        let nested_fields_extended: Fields = vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Utf8, true),
        ]
        .into();
        let arrow_schema_extended = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "nested",
                DataType::Struct(nested_fields_extended.clone()),
                true,
            ),
        ]));
        let parquet_predicate_schema_extended =
            build_parquet_predicate_schema(&arrow_schema_extended, &file_id_field);
        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema_extended,
            &parquet_predicate_schema_extended,
            None,
            &file_id_field,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+--------------------+-----------------------------+",
            "| id | nested             | __delta_rs_file_id__        |",
            "+----+--------------------+-----------------------------+",
            "| 1  | {a: a, b: aa, c: } | memory:///test_data.parquet |",
            "| 2  | {a: b, b: bb, c: } | memory:///test_data.parquet |",
            "| 3  | {a: c, b: cc, c: } | memory:///test_data.parquet |",
            "+----+--------------------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_parquet_plan_multiple_stores() -> TestResult {
        let store_1 = Arc::new(InMemory::new());
        let store_url_1 = Url::parse("first:///")?;
        let store_2 = Arc::new(InMemory::new());
        let store_url_2 = Url::parse("second:///")?;

        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url_1, store_1.clone());
        session
            .runtime_env()
            .register_object_store(&store_url_2, store_2.clone());

        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));

        let data_1 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(StringArray::from(vec![Some("a")])),
            ],
        )?;
        let data_2 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2])),
                Arc::new(StringArray::from(vec![Some("b")])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data_1)?;
        arrow_writer.close()?;
        let path = Path::from("test_data.parquet");
        store_1.put(&path, buffer.into()).await?;
        let mut file_1: PartitionedFile = store_1.head(&path).await?.into();
        file_1
            .partition_values
            .push(wrap_file_id_value("first:///test_data.parquet"));

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data_2)?;
        arrow_writer.close()?;
        let path = Path::from("test_data.parquet");
        store_2.put(&path, buffer.into()).await?;
        let mut file_2: PartitionedFile = store_2.head(&path).await?.into();
        file_2
            .partition_values
            .push(wrap_file_id_value("second:///test_data.parquet"));

        let files_by_store = vec![
            (
                store_url_1.as_object_store_url(),
                vec![(file_1, None::<Vec<bool>>)],
            ),
            (
                store_url_2.as_object_store_url(),
                vec![(file_2, None::<Vec<bool>>)],
            ),
        ];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 1  | a     | first:///test_data.parquet  |",
            "| 2  | b     | second:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_parquet_plan_predicate() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_data.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_data.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let predicate = col("id").eq(lit(2i32));
        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 2  | b     | memory:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_skips_pushdown_when_logical_rewrite_fails() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let parquet_read_schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let logical_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("missing", DataType::Int32, false),
        ]));
        let data = RecordBatch::try_new(
            parquet_read_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer =
            ArrowWriter::try_new(&mut buffer, parquet_read_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_rewrite_failure.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_rewrite_failure.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&logical_schema, &file_id_field);
        let predicate = col("missing").eq(lit(1i32));

        let plan = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+----------------------------------------+",
            "| id | __delta_rs_file_id__                   |",
            "+----+----------------------------------------+",
            "| 1  | memory:///test_rewrite_failure.parquet |",
            "| 2  | memory:///test_rewrite_failure.parquet |",
            "| 3  | memory:///test_rewrite_failure.parquet |",
            "+----+----------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_view_literal_against_base_parquet_file() -> TestResult {
        use datafusion::scalar::ScalarValue;

        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        // Write a Parquet file with base types, but read it with a view-typed schema.
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8View, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    Some("alice"),
                    Some("bob"),
                    Some("charlie"),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_view_literal.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_view_literal.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col("name").eq(lit(ScalarValue::Utf8View(Some("bob".to_string()))));
        let plan = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;

        let expected = vec![
            "+----+------+-------------------------------------+",
            "| id | name | __delta_rs_file_id__                |",
            "+----+------+-------------------------------------+",
            "| 2  | bob  | memory:///test_view_literal.parquet |",
            "+----+------+-------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_sql_literal_against_view_schema() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        // Write a Parquet file with base types, but read it with a view-typed schema.
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8View, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    Some("alice"),
                    Some("bob"),
                    Some("charlie"),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_sql_literal.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_sql_literal.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col("name").eq(lit("bob"));
        let plan = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;

        let expected = vec![
            "+----+------+------------------------------------+",
            "| id | name | __delta_rs_file_id__               |",
            "+----+------+------------------------------------+",
            "| 2  | bob  | memory:///test_sql_literal.parquet |",
            "+----+------+------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_physical_column_mapping_names() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let physical_name = "col-3877fd94-0973-4941-ac6b-646849a1ff65";
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(physical_name, DataType::Utf8, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(physical_name, DataType::Utf8View, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    Some("alice"),
                    Some("bob"),
                    Some("charlie"),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_column_mapping_pushdown.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values.push(wrap_file_id_value(
            "memory:///test_column_mapping_pushdown.parquet",
        ));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col(physical_name).eq(lit("bob"));
        let plan = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[0].num_columns(), 3);

        let id_col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 2);

        let name_col = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        assert_eq!(name_col.value(0), "bob");

        assert_eq!(batches[0].schema().field(1).name(), physical_name);
        assert_eq!(batches[0].schema().field(2).name(), FILE_ID_COLUMN_DEFAULT);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_binaryview_literal_against_base_parquet_file()
    -> TestResult {
        use datafusion::scalar::ScalarValue;

        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Binary, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::BinaryView, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(BinaryArray::from_opt_vec(vec![
                    Some(b"aaa".as_slice()),
                    Some(b"bbb".as_slice()),
                    Some(b"ccc".as_slice()),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_binary_view.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_binary_view.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col("data").eq(lit(ScalarValue::BinaryView(Some(b"bbb".to_vec()))));
        let plan = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        let id_col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 2);

        let data_col = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();
        assert_eq!(data_col.value(0), b"bbb");

        assert_eq!(batches[0].num_columns(), 3);
        assert_eq!(batches[0].schema().field(2).name(), FILE_ID_COLUMN_DEFAULT);

        Ok(())
    }
}
