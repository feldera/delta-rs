//! Asynchronous deletion-vector loading for scan planning.
//!
//! Deletion vectors are read with async object-store calls and decoded on the
//! caller's task. Nothing here blocks a thread.
//!
//! That property is the point. The kernel's `Engine` API is synchronous, so the
//! obvious way to call [`DvInfo::get_selection_vector`] from async code is
//! `spawn_blocking`. But the kernel then bridges back to async internally
//! (`TokioMultiThreadExecutor::block_on`), and that bridge hands its result over
//! through a nested `spawn_blocking`. One deletion vector therefore occupies a
//! blocking thread *and* needs a second one to finish. Load enough of them at
//! once and every thread in Tokio's blocking pool is waiting for a hand-off that
//! can only be scheduled onto that same pool: a deadlock that no amount of pool
//! sizing removes, because the fan-out is one task per file.
//!
//! The fix is to take the I/O out of the synchronous call. [`absolute_path`]
//! says which object holds the vector, so we fetch it ourselves and hand the
//! kernel a [`StorageHandler`] that already has the bytes. The kernel still owns
//! the on-disk format; its `read` just no longer touches storage, which makes it
//! pure CPU and safe to run inline. Concurrency is then bounded with
//! [`buffer_unordered`], which limits futures rather than threads, so no value
//! of the bound can deadlock.
//!
//! [`absolute_path`]: delta_kernel::actions::deletion_vector::DeletionVectorDescriptor::absolute_path
//! [`buffer_unordered`]: futures::StreamExt::buffer_unordered

use std::sync::Arc;

use bytes::Bytes;
use dashmap::DashMap;
use datafusion::execution::runtime_env::RuntimeEnv;
use delta_kernel::{
    DeltaResult as KernelResult, Engine, Error as KernelError, EvaluationHandler, FileMeta,
    FileSlice, JsonHandler, ParquetHandler, StorageHandler,
    actions::deletion_vector::{
        DeletionVectorDescriptor as KernelDvDescriptor, DeletionVectorStorageType,
    },
    scan::state::DvInfo,
};
use futures::{StreamExt as _, TryStreamExt as _};
use object_store::{ObjectStoreExt as _, path::Path};
use tokio::sync::mpsc::UnboundedReceiver;
use url::Url;

use crate::{
    DeltaResult, DeltaTableError,
    delta_datafusion::engine::AsObjectStoreUrl as _,
    kernel::{DeletionVectorDescriptor, StorageType},
};

/// Deletion vectors fetched concurrently during scan planning.
///
/// Bounds in-flight object-store requests, not threads: a queued load costs a
/// future. Sized for object-store throughput rather than for safety, since no
/// value of it can deadlock.
const DV_LOAD_CONCURRENCY: usize = 32;

/// A deletion vector discovered during replay, waiting to be loaded.
pub(crate) struct PendingDv {
    /// The data file the vector applies to. Keys the result map.
    pub(crate) file_url: Url,
    pub(crate) descriptor: DeletionVectorDescriptor,
    /// `numRecords` from the file's statistics, when present.
    pub(crate) num_records: Option<u64>,
}

/// One loaded deletion vector: the file it belongs to, its keep mask, and the
/// file's row count when the statistics carried one.
pub(crate) type LoadedDv = (Url, Option<Vec<bool>>, Option<u64>);

/// Translate delta-rs's descriptor into the kernel's equivalent. The fields are
/// identical; only the storage-type enum differs.
fn to_kernel_descriptor(dv: &DeletionVectorDescriptor) -> KernelDvDescriptor {
    KernelDvDescriptor {
        storage_type: match dv.storage_type {
            StorageType::UuidRelativePath => DeletionVectorStorageType::PersistedRelative,
            StorageType::Inline => DeletionVectorStorageType::Inline,
            StorageType::AbsolutePath => DeletionVectorStorageType::PersistedAbsolute,
        },
        path_or_inline_dv: dv.path_or_inline_dv.clone(),
        offset: dv.offset,
        size_in_bytes: dv.size_in_bytes,
        cardinality: dv.cardinality,
    }
}

/// A [`StorageHandler`] serving one already-fetched deletion-vector file.
///
/// The kernel asks for the whole object (it passes `None` for the range and
/// indexes into the bytes itself), so a single buffer is all it ever needs.
/// Every other operation is unreachable on the deletion-vector path and returns
/// an error rather than pretending to work.
#[derive(Debug)]
struct PrefetchedDv {
    /// `None` for an inline vector, whose bytes live in the descriptor itself
    /// and which therefore never reaches storage.
    file: Option<(Url, Bytes)>,
}

impl PrefetchedDv {
    fn new(url: Url, bytes: Bytes) -> Self {
        Self {
            file: Some((url, bytes)),
        }
    }

    /// For inline vectors: the kernel decodes the descriptor directly.
    fn inline() -> Self {
        Self { file: None }
    }

    fn unsupported<T>(op: &str) -> KernelResult<T> {
        Err(KernelError::generic(format!(
            "PrefetchedDv serves prefetched deletion-vector reads only; {op} is not supported"
        )))
    }
}

impl StorageHandler for PrefetchedDv {
    fn read_files(
        &self,
        files: Vec<FileSlice>,
    ) -> KernelResult<Box<dyn Iterator<Item = KernelResult<Bytes>>>> {
        let results = files
            .into_iter()
            .map(|(url, _range)| match &self.file {
                Some((prefetched, bytes)) if *prefetched == url => Ok(bytes.clone()),
                _ => Err(KernelError::file_not_found(url)),
            })
            .collect::<Vec<_>>();
        Ok(Box::new(results.into_iter()))
    }

    fn list_from(
        &self,
        _path: &Url,
    ) -> KernelResult<Box<dyn Iterator<Item = KernelResult<FileMeta>>>> {
        Self::unsupported("list_from")
    }

    fn copy_atomic(&self, _src: &Url, _dest: &Url) -> KernelResult<()> {
        Self::unsupported("copy_atomic")
    }

    fn put(&self, _path: &Url, _data: Bytes, _overwrite: bool) -> KernelResult<()> {
        Self::unsupported("put")
    }

    fn head(&self, _path: &Url) -> KernelResult<FileMeta> {
        Self::unsupported("head")
    }
}

/// The scan's engine with its storage swapped for prefetched bytes.
///
/// Only `storage_handler` differs; expression, JSON, and Parquet handling stay
/// with the real engine so decoding behaves exactly as it does today.
struct PrefetchedEngine {
    inner: Arc<dyn Engine>,
    storage: Arc<PrefetchedDv>,
}

impl Engine for PrefetchedEngine {
    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        self.storage.clone()
    }

    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        self.inner.evaluation_handler()
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        self.inner.json_handler()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        self.inner.parquet_handler()
    }
}

/// Load one deletion vector: fetch its bytes if it has any, then decode.
async fn load_one(
    runtime: Arc<RuntimeEnv>,
    engine: Arc<dyn Engine>,
    table_root: Url,
    pending: PendingDv,
) -> DeltaResult<LoadedDv> {
    let descriptor = to_kernel_descriptor(&pending.descriptor);

    let storage = match descriptor
        .absolute_path(&table_root)
        .map_err(|e| DeltaTableError::generic(format!("invalid deletion vector path: {e}")))?
    {
        Some(url) => {
            let path = Path::from_url_path(url.path()).map_err(|e| {
                DeltaTableError::generic(format!("invalid deletion vector path {url}: {e}"))
            })?;
            let store = runtime
                .object_store(url.as_object_store_url())
                .map_err(|e| DeltaTableError::generic(format!("no object store for {url}: {e}")))?;
            let bytes = store.get(&path).await?.bytes().await?;
            Arc::new(PrefetchedDv::new(url, bytes))
        }
        None => Arc::new(PrefetchedDv::inline()),
    };

    // No I/O below this line: the bytes are already in memory.
    let engine = PrefetchedEngine {
        inner: engine,
        storage,
    };
    let keep_mask = DvInfo::from(descriptor)
        .get_selection_vector(&engine, &table_root)
        .map_err(|e| DeltaTableError::generic(format!("failed to read deletion vector: {e}")))?;

    Ok((pending.file_url, keep_mask, pending.num_records))
}

/// Load every deletion vector arriving on `pending`, at most
/// [`DV_LOAD_CONCURRENCY`] at a time.
///
/// Returns when the sender is dropped and all in-flight loads have finished, so
/// callers should run this concurrently with the replay that feeds it.
pub(crate) async fn load_deletion_vectors(
    runtime: Arc<RuntimeEnv>,
    engine: Arc<dyn Engine>,
    table_root: Url,
    pending: UnboundedReceiver<PendingDv>,
) -> DeltaResult<Vec<LoadedDv>> {
    futures::stream::unfold(pending, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    })
    .map(|dv| load_one(runtime.clone(), engine.clone(), table_root.clone(), dv))
    .buffer_unordered(DV_LOAD_CONCURRENCY)
    .try_collect()
    .await
}

/// Collect loaded deletion vectors into the keep-mask map the scan plan wants,
/// dropping files whose vector turned out to be absent.
pub(crate) fn into_keep_masks(loaded: Vec<LoadedDv>) -> DashMap<String, Vec<bool>> {
    loaded
        .into_iter()
        .filter_map(|(url, mask, _)| mask.map(|mask| (url.to_string(), mask)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(storage_type: StorageType, path_or_inline_dv: &str) -> DeletionVectorDescriptor {
        DeletionVectorDescriptor {
            storage_type,
            path_or_inline_dv: path_or_inline_dv.to_string(),
            offset: Some(1),
            size_in_bytes: 42,
            cardinality: 1,
        }
    }

    #[test]
    fn kernel_descriptor_preserves_every_field() {
        let dv = descriptor(StorageType::UuidRelativePath, "ab^-aqEH.-t@S}K{vb[*k^");
        let kernel = to_kernel_descriptor(&dv);

        assert_eq!(
            kernel.storage_type,
            DeletionVectorStorageType::PersistedRelative
        );
        assert_eq!(kernel.path_or_inline_dv, dv.path_or_inline_dv);
        assert_eq!(kernel.offset, dv.offset);
        assert_eq!(kernel.size_in_bytes, dv.size_in_bytes);
        assert_eq!(kernel.cardinality, dv.cardinality);
    }

    #[test]
    fn kernel_descriptor_maps_each_storage_type() {
        for (ours, theirs) in [
            (
                StorageType::UuidRelativePath,
                DeletionVectorStorageType::PersistedRelative,
            ),
            (StorageType::Inline, DeletionVectorStorageType::Inline),
            (
                StorageType::AbsolutePath,
                DeletionVectorStorageType::PersistedAbsolute,
            ),
        ] {
            assert_eq!(
                to_kernel_descriptor(&descriptor(ours, "x")).storage_type,
                theirs
            );
        }
    }

    #[test]
    fn prefetched_storage_serves_the_file_it_holds() {
        let url = Url::parse("memory:///deletion_vector_1.bin").unwrap();
        let storage = PrefetchedDv::new(url.clone(), Bytes::from_static(b"payload"));

        let mut read = storage.read_files(vec![(url, None)]).unwrap();
        assert_eq!(
            read.next().unwrap().unwrap(),
            Bytes::from_static(b"payload")
        );
        assert!(read.next().is_none());
    }

    /// A wrong URL must fail rather than silently return the wrong vector,
    /// which would delete the wrong rows.
    #[test]
    fn prefetched_storage_rejects_any_other_file() {
        let storage = PrefetchedDv::new(
            Url::parse("memory:///deletion_vector_1.bin").unwrap(),
            Bytes::from_static(b"payload"),
        );

        let other = Url::parse("memory:///deletion_vector_2.bin").unwrap();
        let mut read = storage.read_files(vec![(other, None)]).unwrap();
        assert!(read.next().unwrap().is_err());
    }

    #[test]
    fn prefetched_storage_refuses_operations_it_cannot_serve() {
        let storage = PrefetchedDv::inline();
        let url = Url::parse("memory:///x").unwrap();

        assert!(storage.list_from(&url).is_err());
        assert!(storage.head(&url).is_err());
        assert!(storage.put(&url, Bytes::new(), true).is_err());
        assert!(storage.copy_atomic(&url, &url).is_err());
    }
}
