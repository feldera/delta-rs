//! Delta-aware physical expression adaptation for parquet scans.

use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::{CastOptions, cast_with_options};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TreeNode as _};
use datafusion::common::{DataFusionError, HashMap, Result, plan_err};
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::{PhysicalExpr, expressions::Column};
use datafusion_physical_expr_adapter::{
    DefaultPhysicalExprAdapterFactory, PhysicalExprAdapter, PhysicalExprAdapterFactory,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

/// A [`PhysicalExprAdapterFactory`] that relaxes nested-field nullability on the
/// logical (table) schema before delegating to DataFusion's default adapter.
///
/// In Delta, `nullable = false` is a write-time invariant: engines such as Spark
/// write parquet files whose nested fields are physically optional even when the
/// table schema declares them non-nullable. DataFusion's default adapter rejects
/// nullable -> non-nullable nested struct field casts at plan time, so we hand it
/// a nullability-relaxed target schema instead. The data is guaranteed non-null
/// by the writer, and the logical output schema is restored above the parquet
/// scan by [`DeltaScanExec`](super::DeltaScanExec)'s transforms.
#[derive(Debug)]
pub(crate) struct DeltaPhysicalExprAdapterFactory;

impl PhysicalExprAdapterFactory for DeltaPhysicalExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Result<Arc<dyn PhysicalExprAdapter>> {
        DefaultPhysicalExprAdapterFactory.create(
            Arc::new(relax_schema_nested_nullability(&logical_file_schema)),
            physical_file_schema,
        )
    }
}

/// Relax nullability of all *nested* fields. Top-level field nullability is kept
/// as is: the default adapter already tolerates nullable file columns feeding
/// non-nullable top-level table columns, and keeping it strict preserves the
/// "non-nullable column missing from file" error for missing top-level columns.
pub(super) fn relax_schema_nested_nullability(schema: &Schema) -> Schema {
    let fields: Vec<FieldRef> = schema
        .fields()
        .iter()
        .map(|f| relax_field_type(f.as_ref()))
        .collect();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

/// Relax nullability within the field's data type, keeping the field's own
/// nullability unchanged.
fn relax_field_type(field: &Field) -> FieldRef {
    Arc::new(
        field
            .clone()
            .with_data_type(relax_nested_nullability(field.data_type())),
    )
}

/// Relax nullability within the field's data type and mark the field itself as
/// nullable.
fn relax_field(field: &FieldRef) -> FieldRef {
    Arc::new(relax_field_type(field).as_ref().clone().with_nullable(true))
}

fn relax_nested_nullability(dt: &DataType) -> DataType {
    match dt {
        DataType::Struct(fields) => DataType::Struct(fields.iter().map(relax_field).collect()),
        DataType::List(f) => DataType::List(relax_field(f)),
        DataType::LargeList(f) => DataType::LargeList(relax_field(f)),
        DataType::ListView(f) => DataType::ListView(relax_field(f)),
        DataType::LargeListView(f) => DataType::LargeListView(relax_field(f)),
        DataType::FixedSizeList(f, n) => DataType::FixedSizeList(relax_field(f), *n),
        DataType::Map(entries, sorted) => {
            // Arrow requires map entries and keys to be non-nullable; only the
            // value field may be relaxed.
            let entries_type = match entries.data_type() {
                DataType::Struct(kv) if kv.len() == 2 => DataType::Struct(Fields::from(vec![
                    relax_field_type(kv[0].as_ref()),
                    relax_field(&kv[1]),
                ])),
                other => relax_nested_nullability(other),
            };
            DataType::Map(
                Arc::new(entries.as_ref().clone().with_data_type(entries_type)),
                *sorted,
            )
        }
        DataType::Dictionary(k, v) => {
            DataType::Dictionary(k.clone(), Box::new(relax_nested_nullability(v)))
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relax_keeps_top_level_nullability_and_relaxes_nested_fields() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "meta",
                DataType::Struct(Fields::from(vec![Field::new(
                    "int_id",
                    DataType::Utf8,
                    false,
                )])),
                true,
            ),
        ]);

        let relaxed = relax_schema_nested_nullability(&schema);

        // Top-level nullability is preserved.
        assert!(!relaxed.field(0).is_nullable());
        assert!(relaxed.field(1).is_nullable());

        // Nested struct field is relaxed to nullable.
        let DataType::Struct(fields) = relaxed.field(1).data_type() else {
            panic!("expected struct");
        };
        assert!(fields[0].is_nullable());
    }

    #[test]
    fn relax_map_keeps_entries_and_key_non_nullable() {
        let entries = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("keys", DataType::Utf8, false),
                Field::new(
                    "values",
                    DataType::Struct(Fields::from(vec![Field::new("v", DataType::Int32, false)])),
                    false,
                ),
            ])),
            false,
        );
        let schema = Schema::new(vec![Field::new(
            "m",
            DataType::Map(Arc::new(entries), false),
            true,
        )]);

        let relaxed = relax_schema_nested_nullability(&schema);
        let DataType::Map(entries, _) = relaxed.field(0).data_type() else {
            panic!("expected map");
        };
        assert!(!entries.is_nullable());
        let DataType::Struct(kv) = entries.data_type() else {
            panic!("expected struct entries");
        };
        assert!(!kv[0].is_nullable(), "map keys must stay non-nullable");
        assert!(kv[1].is_nullable(), "map values should be relaxed");
        let DataType::Struct(value_fields) = kv[1].data_type() else {
            panic!("expected struct values");
        };
        assert!(value_fields[0].is_nullable());
    }

    #[test]
    fn relax_list_variants_relax_element_nullability() {
        let elem = Arc::new(Field::new("item", DataType::Int32, false));
        let schema = Schema::new(vec![
            Field::new("large", DataType::LargeList(elem.clone()), true),
            Field::new("view", DataType::ListView(elem.clone()), true),
            Field::new("large_view", DataType::LargeListView(elem.clone()), true),
            Field::new("fixed", DataType::FixedSizeList(elem.clone(), 2), true),
        ]);

        let relaxed = relax_schema_nested_nullability(&schema);

        for field in relaxed.fields() {
            let inner = match field.data_type() {
                DataType::LargeList(f) | DataType::ListView(f) | DataType::LargeListView(f) => f,
                DataType::FixedSizeList(f, n) => {
                    assert_eq!(*n, 2, "list size must be preserved");
                    f
                }
                other => panic!("unexpected data type: {other:?}"),
            };
            assert!(
                inner.is_nullable(),
                "element of {} should be relaxed",
                field.name()
            );
        }
    }

    #[test]
    fn relax_map_with_non_standard_entries_falls_back_to_generic_relaxation() {
        // Entries struct with three fields does not match the key/value shape,
        // so relaxation falls back to the generic nested handling.
        let entries = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Utf8, false),
                Field::new("b", DataType::Int32, false),
                Field::new("c", DataType::Int32, false),
            ])),
            false,
        );
        let schema = Schema::new(vec![Field::new(
            "m",
            DataType::Map(Arc::new(entries), false),
            true,
        )]);

        let relaxed = relax_schema_nested_nullability(&schema);
        let DataType::Map(entries, _) = relaxed.field(0).data_type() else {
            panic!("expected map");
        };
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("expected struct entries");
        };
        assert!(fields.iter().all(|f| f.is_nullable()));
    }
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
pub(crate) struct FieldIdAlignedExprAdapterFactory {
    inner: DeltaPhysicalExprAdapterFactory,
}

impl Default for FieldIdAlignedExprAdapterFactory {
    fn default() -> Self {
        Self {
            inner: DeltaPhysicalExprAdapterFactory,
        }
    }
}

impl PhysicalExprAdapterFactory for FieldIdAlignedExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Result<Arc<dyn PhysicalExprAdapter>> {
        let alignment =
            align_read_schema_to_file_names(&logical_file_schema, &physical_file_schema);
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
            .transform_down(|node| {
                match (node.as_ref() as &dyn std::any::Any).downcast_ref::<Column>() {
                    Some(column) => {
                        let file_name = self
                            .read_to_file
                            .get(column.name())
                            .map(String::as_str)
                            .unwrap_or_else(|| column.name());
                        let file_column: Arc<dyn PhysicalExpr> =
                            Arc::new(Column::new(file_name, column.index()));
                        match self.realign_targets.get(column.name()) {
                            Some(target) => Ok(Transformed::yes(Arc::new(
                                FieldIdRealignExpr::new(file_column, target.data_type().clone()),
                            ))),
                            None if file_name != column.name() => Ok(Transformed::yes(file_column)),
                            None => Ok(Transformed::no(node)),
                        }
                    }
                    None => Ok(Transformed::no(node)),
                }
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
/// The element field of a `List` or a `LargeList`, the two flavors a Delta array
/// arrives as; `None` for any other type.
fn list_element(data_type: &DataType) -> Option<&FieldRef> {
    match data_type {
        DataType::List(inner) | DataType::LargeList(inner) => Some(inner),
        _ => None,
    }
}

/// A map's `entries`, `key` and `value` fields relabeled to the file's names.
///
/// Delta assigns a column mapping id to the map itself, not to the three fields
/// that spell out its layout, so [`align_type_to_file_names`]'s struct arm finds no
/// id to pair them on. Both sides always hold a key and a value in that order, so
/// pair them by position and recurse, which reaches the ids the value's own struct
/// children do carry.
fn align_map_entries(read: &FieldRef, file: &FieldRef) -> Field {
    let aligned = match (read.data_type(), file.data_type()) {
        (DataType::Struct(read_children), DataType::Struct(file_children))
            if read_children.len() == file_children.len() =>
        {
            let children: Vec<FieldRef> = read_children
                .iter()
                .zip(file_children)
                .map(|(read_child, file_child)| {
                    Arc::new(align_field_to_file(read_child, file_child))
                })
                .collect();
            DataType::Struct(children.into())
        }
        _ => read.data_type().clone(),
    };
    read.as_ref()
        .clone()
        .with_name(file.name())
        .with_data_type(aligned)
}

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
        // The log and the file need not agree on the list flavor, so match on any.
        (List(read_inner), _) => match list_element(file) {
            Some(file_inner) => List(Arc::new(align_field_to_file(read_inner, file_inner))),
            None => read.clone(),
        },
        (LargeList(read_inner), _) => match list_element(file) {
            Some(file_inner) => LargeList(Arc::new(align_field_to_file(read_inner, file_inner))),
            None => read.clone(),
        },
        (Map(read_entries, sorted), Map(file_entries, _)) => Map(
            Arc::new(align_map_entries(read_entries, file_entries)),
            *sorted,
        ),
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

#[cfg(test)]
mod field_id_tests {
    use std::sync::Arc;

    use arrow::array::{Array, ArrayRef, StringArray, StringViewArray, StructArray};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    use super::*;

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
            vec![Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])) as ArrayRef],
            None,
        )
        .unwrap();
        let source_inner = Field::new("item", structs.data_type().clone(), true)
            .with_metadata(field_id_meta("103"));
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
                vec![
                    Field::new("col-104", DataType::Utf8View, true)
                        .with_metadata(field_id_meta("104")),
                ]
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
        let items = out.values().as_any().downcast_ref::<StructArray>().unwrap();
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
}
