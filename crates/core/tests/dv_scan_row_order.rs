//! A deletion-vector scan must return the rows the vectors keep, whatever the
//! planner does underneath it.
//!
//! `DeltaScanStream` consumes each file's keep mask positionally: the first
//! rows of a file take the head of the mask. That holds only while the file
//! arrives whole, in physical row order, with every row still present.
//! `DeltaScanExec` asks for a single input partition, which is not enough --
//! `EnforceDistribution` satisfies the request by merging split partitions with
//! a `CoalescePartitionsExec`, and a merge is not ordered.
//!
//! Both tests need a table Delta Spark wrote, because delta-rs cannot write
//! deletion vectors. Build it with `big_dv_fixture.py` / `mid_dv_fixture.py`
//! and point the variables below at the result; they skip when unset.
//!
//! The row *count* stays correct in every failing case here. Only the identity
//! of the rows is wrong, so assert on the deleted rows, never on the count.

use datafusion::arrow::array::Int64Array;
use datafusion::prelude::{SessionConfig, SessionContext};
use std::sync::Arc;

async fn counts(path: &str, sql: &str, split_files: bool, partitions: usize) -> (i64, i64, String) {
    let url = url::Url::from_directory_path(path).unwrap();
    let table = deltalake_core::open_table(url).await.unwrap();
    let provider = table.table_provider().await.unwrap();

    let mut config = SessionConfig::new().with_target_partitions(partitions);
    config.options_mut().optimizer.repartition_file_scans = split_files;
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("t", Arc::clone(&provider)).unwrap();

    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let rendered = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();

    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let value = |index: usize| {
        batches[0]
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    };
    (value(0), value(1), rendered)
}

/// Byte-range splitting hands one file to several scan partitions, which are
/// then merged in completion order.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn deletion_vectors_survive_file_splitting() {
    let Ok(path) = std::env::var("DV_SPLIT_TABLE") else {
        eprintln!("DV_SPLIT_TABLE unset; skipping");
        return;
    };
    const ROWS: i64 = 2_000_000;
    const HALF: i64 = ROWS / 2;
    const DELETED: i64 = 4978 + 43;
    let sql = format!(
        "SELECT count(*) AS total, count(*) FILTER (WHERE id < 4978 \
         OR (id >= {HALF} AND id < {})) AS resurrected FROM t",
        HALF + 43
    );

    // Splitting off first: it establishes that the fixture and the mask agree,
    // so a failure with splitting on is the split and not the fixture.
    let (total, resurrected, _) = counts(&path, &sql, false, 8).await;
    assert_eq!(
        (total, resurrected),
        (ROWS - DELETED, 0),
        "unsplit read is already wrong; the fixture, not the split, is at fault"
    );

    let (total, resurrected, plan) = counts(&path, &sql, true, 8).await;
    assert!(
        plan.contains("deletion_vector_row_order=preserved"),
        "the whole-file guard is not installed on this scan:\n{plan}"
    );
    assert_eq!(
        resurrected, 0,
        "{resurrected} deleted rows came back once the files were split, with \
         the row count still {total}. The keep mask was applied at the wrong \
         offsets.\n{plan}"
    );
}

/// A pushed-down filter drops rows inside the Parquet reader, beneath the mask.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn deletion_vectors_survive_row_group_pruning() {
    let Ok(path) = std::env::var("DV_MID_TABLE") else {
        eprintln!("DV_MID_TABLE unset; skipping");
        return;
    };
    // Deleted rows sit at 500_000..505_000, mid-file. The filter keeps them in
    // scope but prunes the row groups before them, so the surviving rows no
    // longer begin at the head of the file the mask is aligned to.
    let sql = "SELECT count(*) AS total, \
               count(*) FILTER (WHERE id >= 500000 AND id < 505000) AS resurrected \
               FROM t WHERE id >= 400000 AND id < 600000";
    let (total, resurrected, plan) = counts(&path, sql, false, 8).await;
    assert!(
        plan.contains("deletion_vector_row_order=preserved"),
        "the whole-file guard is not installed on this scan:\n{plan}"
    );
    assert!(
        !plan.contains("pruning_predicate"),
        "a pruning predicate reached the reader:\n{plan}"
    );
    assert_eq!(
        resurrected, 0,
        "{resurrected} deleted rows came back through a pruned scan, total \
         {total}. The predicate reached the Parquet reader under the mask.\n{plan}"
    );
}

/// The guard is scoped to scans that carry deletion vectors: a table without
/// them must still split and prune, or the fix has cost every Delta read.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn tables_without_deletion_vectors_keep_their_optimisations() {
    let Ok(path) = std::env::var("NODV_TABLE") else {
        eprintln!("NODV_TABLE unset; skipping");
        return;
    };
    // `count(*)` on a table with no deletion vectors is answered from the
    // log's statistics -- `PlaceholderRowExec`, no scan to inspect. Sum a
    // column so the plan actually contains one.
    let (_, _, plan) = counts(&path, "SELECT sum(id) AS a, sum(id) AS b FROM t", true, 8).await;
    assert!(
        !plan.contains("deletion_vector_row_order=preserved"),
        "the whole-file guard was applied to a table with no deletion vectors:\n{plan}"
    );
    assert!(
        plan.contains("file_groups={8 groups"),
        "a table without deletion vectors stopped splitting:\n{plan}"
    );
    let (_, _, plan) = counts(
        &path,
        "SELECT sum(id) AS a, sum(id) AS b FROM t WHERE id >= 1999000",
        true,
        8,
    )
    .await;
    assert!(
        plan.contains("pruning_predicate"),
        "a table without deletion vectors stopped pruning:\n{plan}"
    );
}

/// A `LIMIT` must be honoured on a deletion-vector table.
///
/// `DeltaScanMetaExec` answers some queries from file metadata alone. It is a
/// leaf, yet reports `supports_limit_pushdown() == true` while `with_fetch`
/// returns `None`, so DataFusion's `LimitPushdown` removes the `LimitExec`
/// (limit_pushdown.rs:163) and descends looking for something to absorb the
/// fetch. Nothing below a leaf can. This checks the rows that come back.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn limit_is_honoured_on_a_deletion_vector_table() {
    let Ok(path) = std::env::var("DV_SPLIT_TABLE") else {
        eprintln!("DV_SPLIT_TABLE unset; skipping");
        return;
    };
    for sql in [
        "SELECT count(*) AS a, count(*) AS b FROM (SELECT id FROM t LIMIT 5)",
        "SELECT count(*) AS a, count(*) AS b FROM (SELECT id FROM t WHERE id > 10 LIMIT 5)",
    ] {
        let (rows, _, plan) = counts(&path, sql, false, 8).await;
        assert_eq!(rows, 5, "LIMIT 5 returned {rows} rows for `{sql}`\n{plan}");
    }
}
