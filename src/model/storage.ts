/**
 * Trained models, kept in the browser so they survive a reload.
 *
 * Training takes minutes, so losing the result to a refresh is not acceptable, and a page has
 * nowhere else to put a couple of megabytes. IndexedDB is per-origin and per-device: a model saved
 * here is private to this browser, and sharing one means downloading the `.dnw` and sending it, or
 * committing it to `public/models/` so it ships with the deployed build.
 *
 * Metadata and weights live in separate stores so the model list can be read without pulling every
 * model's megabytes into memory to render a dropdown.
 */

const DATABASE_NAME = 'dreamnet';
const DATABASE_VERSION = 1;
const META_STORE = 'model-meta';
const BLOB_STORE = 'model-blobs';

export interface SavedModelMeta {
  id: string;
  name: string;
  description: string;
  bytes: number;
  savedAt: number;
  trainedAt?: number;
  /** Slider labels, so the list can say what a model offers without loading it. */
  controls: string[];
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(META_STORE)) {
        database.createObjectStore(META_STORE, { keyPath: 'id' });
      }
      if (!database.objectStoreNames.contains(BLOB_STORE)) {
        database.createObjectStore(BLOB_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Could not open the model database.'));
  });
}

function finish(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error('The model database transaction failed.'));
    transaction.onabort = () => reject(transaction.error ?? new Error('The model database transaction was aborted.'));
  });
}

export async function saveModel(meta: Omit<SavedModelMeta, 'id' | 'savedAt'>, data: ArrayBuffer): Promise<SavedModelMeta> {
  const record: SavedModelMeta = {
    ...meta,
    id: `local-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    savedAt: Date.now(),
  };

  const database = await openDatabase();
  try {
    const transaction = database.transaction([META_STORE, BLOB_STORE], 'readwrite');
    transaction.objectStore(META_STORE).put(record);
    transaction.objectStore(BLOB_STORE).put(data, record.id);
    await finish(transaction);
  } finally {
    database.close();
  }

  return record;
}

export async function listSavedModels(): Promise<SavedModelMeta[]> {
  let database: IDBDatabase;
  try {
    database = await openDatabase();
  } catch {
    // Private browsing and some locked-down configurations refuse IndexedDB outright. Saved models
    // are a convenience, not a requirement, so this reports "none" rather than breaking the app.
    return [];
  }

  try {
    const store = database.transaction(META_STORE, 'readonly').objectStore(META_STORE);
    const models = await new Promise<SavedModelMeta[]>((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result as SavedModelMeta[]);
      request.onerror = () => reject(request.error);
    });
    return models.sort((a, b) => b.savedAt - a.savedAt);
  } finally {
    database.close();
  }
}

export async function loadSavedModel(id: string): Promise<ArrayBuffer> {
  const database = await openDatabase();
  try {
    const store = database.transaction(BLOB_STORE, 'readonly').objectStore(BLOB_STORE);
    const data = await new Promise<ArrayBuffer | undefined>((resolve, reject) => {
      const request = store.get(id);
      request.onsuccess = () => resolve(request.result as ArrayBuffer | undefined);
      request.onerror = () => reject(request.error);
    });
    if (!data) throw new Error(`Saved model "${id}" is missing its weights.`);
    return data;
  } finally {
    database.close();
  }
}

export async function deleteSavedModel(id: string): Promise<void> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction([META_STORE, BLOB_STORE], 'readwrite');
    transaction.objectStore(META_STORE).delete(id);
    transaction.objectStore(BLOB_STORE).delete(id);
    await finish(transaction);
  } finally {
    database.close();
  }
}
