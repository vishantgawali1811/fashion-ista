(() => {
  const STORAGE_KEY = "threadline-retail-stock-v1";
  const DATABASE_NAME = "threadline-retail-catalog";
  const STORE_NAME = "product-photos";
  let databasePromise;

  function openDatabase() {
    if (!("indexedDB" in window)) {
      return Promise.reject(new Error("This browser does not support local product photo storage."));
    }
    if (!databasePromise) {
      databasePromise = new Promise((resolve, reject) => {
        const request = indexedDB.open(DATABASE_NAME, 1);
        request.onupgradeneeded = () => {
          const database = request.result;
          if (!database.objectStoreNames.contains(STORE_NAME)) {
            database.createObjectStore(STORE_NAME);
          }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error("Could not open product photo storage."));
        request.onblocked = () => reject(new Error("Product photo storage is blocked by another tab."));
      });
    }
    return databasePromise;
  }

  async function imageRequest(mode, imageId, image) {
    const database = await openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, mode);
      const store = transaction.objectStore(STORE_NAME);
      const request =
        mode === "readwrite" && image !== undefined
          ? store.put(image, imageId)
          : mode === "readwrite"
            ? store.delete(imageId)
            : store.get(imageId);
      let result;
      request.onsuccess = () => {
        result = request.result;
      };
      request.onerror = () => reject(request.error ?? new Error("Could not access the product photo."));
      transaction.oncomplete = () => resolve(result);
      transaction.onerror = () =>
        reject(transaction.error ?? new Error("Could not save the product photo."));
      transaction.onabort = () =>
        reject(transaction.error ?? new Error("The product photo transaction was cancelled."));
    });
  }

  function loadEntries() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    const entries = JSON.parse(stored);
    if (
      !Array.isArray(entries) ||
      entries.some(
        (entry) =>
          !entry ||
          typeof entry.id !== "string" ||
          typeof entry.label !== "string" ||
          typeof entry.confidence !== "number" ||
          !Number.isFinite(entry.confidence) ||
          entry.confidence < 0 ||
          entry.confidence > 1 ||
          !Number.isInteger(entry.units) ||
          entry.units < 1 ||
          typeof entry.createdAt !== "string" ||
          !Number.isFinite(Date.parse(entry.createdAt)),
      )
    ) {
      throw new Error("The saved stock log has an unexpected format.");
    }
    return entries.map((entry) => ({
      ...entry,
      reference: typeof entry.reference === "string" ? entry.reference : "",
      productName: typeof entry.productName === "string" ? entry.productName : entry.label,
      variant: typeof entry.variant === "string" ? entry.variant : "",
      price:
        typeof entry.price === "number" && Number.isFinite(entry.price) && entry.price >= 0
          ? entry.price
          : null,
      imageId: typeof entry.imageId === "string" ? entry.imageId : "",
      needsReview: entry.needsReview === true,
      reviewed: entry.reviewed === true,
    }));
  }

  function saveEntries(entries) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  }

  window.ThreadlineCatalog = Object.freeze({
    loadEntries,
    saveEntries,
    saveImage: (imageId, image) => imageRequest("readwrite", imageId, image),
    getImage: (imageId) => imageRequest("readonly", imageId),
    removeImage: (imageId) => imageRequest("readwrite", imageId),
  });
})();
