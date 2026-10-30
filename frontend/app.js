(() => {
  const MAX_FILE_SIZE = 10 * 1024 * 1024;
  const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
  const input = document.querySelector("#image-input");
  const dropzone = document.querySelector("#dropzone");
  const preview = document.querySelector("#drop-preview");
  const previewImage = document.querySelector("#preview-image");
  const selectedFile = document.querySelector("#selected-file");
  const clearButton = document.querySelector("#clear-button");
  const analyzeButton = document.querySelector("#analyze-button");
  const buttonLabel = document.querySelector(".button-label");
  const formMessage = document.querySelector("#form-message");
  const resultStatus = document.querySelector("#result-status");
  const resultEmpty = document.querySelector("#result-empty");
  const resultContent = document.querySelector("#result-content");
  const resultError = document.querySelector("#result-error");
  const errorText = document.querySelector("#error-text");
  const tryAgainButton = document.querySelector("#try-again-button");
  const predictionName = document.querySelector("#prediction-name");
  const confidenceValue = document.querySelector("#confidence-value");
  const confidenceBar = document.querySelector("#confidence-bar");
  const alternatives = document.querySelector("#alternatives");
  const addStockButton = document.querySelector("#add-stock-button");
  const productName = document.querySelector("#product-name");
  const productPrice = document.querySelector("#product-price");
  const productVariant = document.querySelector("#product-variant");
  const productCategory = document.querySelector("#product-category");
  const slotReference = document.querySelector("#slot-reference");
  const unitCount = document.querySelector("#unit-count");
  const stockSearch = document.querySelector("#stock-search");
  const stockRows = document.querySelector("#stock-rows");
  const stockEmpty = document.querySelector("#stock-empty");
  const noSearchResults = document.querySelector("#no-search-results");
  const storageNotice = document.querySelector("#storage-notice");

  let selectedImage = null;
  let previewUrl = null;
  let currentPrediction = null;
  let stockEntries = [];
  let storageAvailable = true;

  function setStorageMessage(message, isError = false) {
    storageNotice.textContent = message;
    storageNotice.classList.toggle("is-error", isError);
  }

  function loadStockEntries() {
    try {
      return window.ThreadlineCatalog.loadEntries();
    } catch (error) {
      storageAvailable = false;
      setStorageMessage(
        `Could not load the saved stock log: ${error.message} Refresh after checking browser storage.`,
        true,
      );
      return [];
    }
  }

  function showMessage(message, isError = false) {
    formMessage.textContent = message;
    formMessage.classList.toggle("is-error", isError);
  }

  function formatFileSize(bytes) {
    return bytes < 1024 * 1024
      ? `${Math.max(1, Math.round(bytes / 1024))} KB`
      : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function setResultState(state) {
    if (state !== "success") {
      currentPrediction = null;
      addStockButton.disabled = true;
    }
    resultEmpty.hidden = state !== "empty";
    resultContent.hidden = state !== "success";
    resultError.hidden = state !== "error";
    resultStatus.classList.toggle("is-ready", state === "success");
    resultStatus.classList.toggle("is-error", state === "error");
    resultStatus.innerHTML =
      state === "success"
        ? "<i></i> ANALYSIS COMPLETE"
        : state === "error"
          ? "<i></i> NEEDS ATTENTION"
          : "<i></i> WAITING FOR IMAGE";
  }

  function setLoading(loading) {
    analyzeButton.disabled = loading || !selectedImage;
    analyzeButton.classList.toggle("is-loading", loading);
    buttonLabel.textContent = loading ? "Sorting item…" : "Suggest a category";
    input.disabled = loading;
    clearButton.disabled = loading;
  }

  function resetResult() {
    setResultState("empty");
    predictionName.textContent = "—";
    productCategory.value = "";
    confidenceValue.textContent = "—";
    confidenceBar.style.width = "0%";
    alternatives.replaceChildren();
  }

  function clearImage() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    selectedImage = null;
    input.value = "";
    previewImage.removeAttribute("src");
    preview.hidden = true;
    selectedFile.hidden = true;
    selectedFile.textContent = "";
    dropzone.classList.remove("has-image");
    clearButton.hidden = true;
    analyzeButton.disabled = true;
    showMessage("");
    productName.value = "";
    productPrice.value = "";
    productVariant.value = "";
    resetResult();
  }

  function acceptImage(file) {
    showMessage("");
    if (!file) return;
    if (!ALLOWED_TYPES.has(file.type)) {
      input.value = "";
      showMessage("Choose a JPG, PNG or WEBP image.", true);
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      input.value = "";
      showMessage("That image is over 10 MB. Choose a smaller file.", true);
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    selectedImage = file;
    previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    preview.hidden = false;
    selectedFile.textContent = `${file.name} · ${formatFileSize(file.size)}`;
    selectedFile.hidden = false;
    clearButton.hidden = false;
    dropzone.classList.add("has-image");
    analyzeButton.disabled = false;
    resetResult();
  }

  function normalizePrediction(data) {
    const label =
      data?.predicted_class ?? data?.label ?? data?.prediction ?? data?.class_name;
    const rawConfidence =
      data?.confidence ?? data?.predicted_probability ?? data?.score;

    if (typeof label !== "string" || label.trim() === "") {
      throw new Error("The prediction service returned an invalid response.");
    }
    const confidence = Number(rawConfidence);
    if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
      throw new Error("The prediction service returned an invalid confidence score.");
    }

    const candidates = Array.isArray(data.top_predictions)
      ? data.top_predictions
      : Array.isArray(data.alternatives)
        ? data.alternatives
        : data.probabilities && typeof data.probabilities === "object"
          ? Object.entries(data.probabilities).map(([name, score]) => ({
              label: name,
              confidence: score,
            }))
          : [];

    const ranked = candidates
      .map((item) => {
        const name = typeof item === "string" ? item : item?.label ?? item?.class_name;
        const score = typeof item === "string" ? null : item?.confidence ?? item?.probability ?? item?.score;
        const value = score === null ? null : Number(score);
        return {
          label: name,
          confidence: Number.isFinite(value) && value >= 0 && value <= 1 ? value : null,
        };
      })
      .filter((item) => typeof item.label === "string" && item.label !== label)
      .sort((a, b) => (b.confidence ?? -1) - (a.confidence ?? -1))
      .slice(0, 3);

    return { label: label.trim(), confidence, ranked };
  }

  function renderPrediction(prediction) {
    currentPrediction = prediction;
    addStockButton.disabled = false;
    productCategory.value = prediction.label;
    predictionName.textContent = prediction.label;
    confidenceValue.textContent = formatPercent(prediction.confidence);
    confidenceBar.style.width = `${Math.round(prediction.confidence * 100)}%`;
    alternatives.replaceChildren();

    if (prediction.ranked.length === 0) {
      const note = document.createElement("p");
      note.className = "no-alternatives";
      note.textContent = "No alternative category scores were provided.";
      alternatives.append(note);
    } else {
      for (const item of prediction.ranked) {
        const row = document.createElement("div");
        row.className = "alternative-row";
        const name = document.createElement("span");
        name.textContent = item.label;
        row.append(name);
        if (item.confidence !== null) {
          const score = document.createElement("strong");
          score.textContent = formatPercent(item.confidence);
          row.append(score);
        }
        alternatives.append(row);
      }
    }
    setResultState("success");
  }

  function formatPercent(value) {
    const percentage = value * 100;
    return percentage > 0 && percentage < 1
      ? "<1%"
      : `${Math.round(percentage)}%`;
  }

  function localDateKey(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  function renderStockLog() {
    const searchTerm = stockSearch.value.trim().toLocaleLowerCase();
    const sorted = [...stockEntries].sort(
      (a, b) => new Date(b.createdAt) - new Date(a.createdAt),
    );
    const visibleEntries = sorted.filter((entry) =>
      `${entry.productName} ${entry.label} ${entry.variant} ${entry.reference}`
        .toLocaleLowerCase()
        .includes(searchTerm),
    );
    stockRows.replaceChildren();

    for (const entry of visibleEntries) {
      const row = document.createElement("tr");
      const category = document.createElement("td");
      category.className = "stock-category";
      const name = document.createElement("strong");
      name.textContent = entry.productName || entry.label;
      category.append(name);
      if (entry.productName && entry.productName !== entry.label) {
        const label = document.createElement("small");
        label.textContent = entry.label;
        category.append(label);
      }
      row.append(category);

      const reference = document.createElement("td");
      reference.textContent = entry.reference || "—";
      row.append(reference);

      const units = document.createElement("td");
      units.className = "stock-count";
      units.textContent = new Intl.NumberFormat().format(entry.units);
      row.append(units);

      const score = document.createElement("td");
      score.className = "stock-price";
      score.textContent =
        typeof entry.price === "number"
          ? new Intl.NumberFormat("en-IN", {
              style: "currency",
              currency: "INR",
              maximumFractionDigits: 0,
            }).format(entry.price)
          : "—";
      row.append(score);

      const match = document.createElement("td");
      match.className = "stock-score";
      match.textContent = formatPercent(entry.confidence);
      row.append(match);

      const statusCell = document.createElement("td");
      const needsReview = entry.needsReview && !entry.reviewed;
      const status = document.createElement("span");
      status.className = `stock-status${needsReview ? " needs-review" : ""}`;
      status.textContent = needsReview ? "Needs check" : "Published to shop";
      statusCell.append(status);
      row.append(statusCell);

      const actions = document.createElement("td");
      if (needsReview) {
        const reviewButton = document.createElement("button");
        reviewButton.type = "button";
        reviewButton.className = "remove-entry review-entry";
        reviewButton.dataset.entryId = entry.id;
        reviewButton.textContent = "Mark checked";
        actions.append(reviewButton);
      }
      const removeButton = document.createElement("button");
      removeButton.type = "button";
      removeButton.className = "remove-entry";
      removeButton.dataset.entryId = entry.id;
      removeButton.setAttribute("aria-label", `Remove ${entry.label} entry`);
      removeButton.textContent = "Remove";
      actions.append(removeButton);
      row.append(actions);
      stockRows.append(row);
    }

    stockEmpty.hidden = stockEntries.length > 0;
    noSearchResults.hidden = stockEntries.length === 0 || visibleEntries.length > 0;

    const todayKey = localDateKey(new Date());
    const todaysEntries = stockEntries.filter(
      (entry) => localDateKey(new Date(entry.createdAt)) === todayKey,
    );
    document.querySelector("#stat-units").textContent = new Intl.NumberFormat().format(
      todaysEntries.reduce((total, entry) => total + entry.units, 0),
    );
    document.querySelector("#stat-categories").textContent = new Set(
      todaysEntries.map((entry) => entry.label),
    ).size;
    document.querySelector("#stat-review").textContent = stockEntries.filter(
      (entry) => entry.needsReview && !entry.reviewed,
    ).length;
    document.querySelector("#stat-live-products").textContent = stockEntries.filter(
      (entry) => entry.units > 0 && (!entry.needsReview || entry.reviewed),
    ).length;
  }

  function persistStockLog(entries) {
    window.ThreadlineCatalog.saveEntries(entries);
  }

  async function addCurrentItemToStock() {
    if (!currentPrediction) {
      showMessage("Get a category suggestion before adding an item to the stock log.", true);
      return;
    }
    if (!storageAvailable) {
      showMessage("The browser stock log is unavailable. Check the storage notice before continuing.", true);
      return;
    }

    const units = Number(unitCount.value);
    if (!Number.isInteger(units) || units < 1 || units > 9999) {
      showMessage("Enter a whole number of pieces between 1 and 9,999.", true);
      unitCount.focus();
      return;
    }

    const label = productCategory.value.trim();
    const name = productName.value.trim();
    const rawPrice = productPrice.value.trim();
    const price = rawPrice === "" ? null : Number(rawPrice);
    if (!label) {
      showMessage("Confirm or enter a product category before adding it.", true);
      productCategory.focus();
      return;
    }
    if (!name) {
      showMessage("Enter a product name so customers can recognize it.", true);
      productName.focus();
      return;
    }
    if (price !== null && (!Number.isFinite(price) || price < 0 || price > 10000000)) {
      showMessage("Enter a valid selling price, or leave it blank.", true);
      productPrice.focus();
      return;
    }

    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    const entry = {
      id,
      label,
      productName: name,
      variant: productVariant.value.trim(),
      price,
      confidence: currentPrediction.confidence,
      units,
      reference: slotReference.value.trim(),
      imageId: id,
      createdAt: new Date().toISOString(),
      needsReview:
        currentPrediction.confidence < 0.65 ||
        label.toLocaleLowerCase().includes("unclear") ||
        label.toLocaleLowerCase().startsWith("other"),
      reviewed: false,
    };
    const nextEntries = [...stockEntries, entry];

    try {
      if (selectedImage) {
        await window.ThreadlineCatalog.saveImage(id, selectedImage);
      }
      persistStockLog(nextEntries);
    } catch (error) {
      try {
        await window.ThreadlineCatalog.removeImage(id);
      } catch (cleanupError) {
        console.error("Could not clean up the unsaved product photo.", cleanupError);
      }
      storageAvailable = false;
      setStorageMessage(`Could not save this entry: ${error.message}`, true);
      showMessage("This item was not added. Check browser photo/storage permissions and try again.", true);
      return;
    }

    stockEntries = nextEntries;
    renderStockLog();
    setStorageMessage("Stock log is saved in this browser on this device.");
    slotReference.value = "";
    unitCount.value = "1";
    clearImage();
    showMessage(`${name} added to your founder stock desk and customer shop.`);
  }

  function updateStockEntry(event) {
    const button = event.target.closest("button[data-entry-id]");
    if (!button || !storageAvailable) return;
    const entryId = button.dataset.entryId;
    const removedEntry = stockEntries.find((entry) => entry.id === entryId);
    let nextEntries;
    if (button.classList.contains("review-entry")) {
      nextEntries = stockEntries.map((entry) =>
        entry.id === entryId ? { ...entry, reviewed: true } : entry,
      );
    } else {
      nextEntries = stockEntries.filter((entry) => entry.id !== entryId);
    }

    try {
      persistStockLog(nextEntries);
    } catch (error) {
      storageAvailable = false;
      setStorageMessage(`Could not update the stock log: ${error.message}`, true);
      return;
    }
    stockEntries = nextEntries;
    renderStockLog();
    if (!button.classList.contains("review-entry")) {
      window.ThreadlineCatalog.removeImage(
        removedEntry?.imageId ?? entryId,
      ).catch((error) => {
        console.error("Could not remove the product photo from local browser storage.", error);
        setStorageMessage("Stock entry removed, but its saved photo could not be removed.", true);
      });
    }
  }

  async function analyzeImage() {
    if (!selectedImage) {
      showMessage("Choose an image before starting the analysis.", true);
      return;
    }

    const predictionUrl = window.THREADLINE_CONFIG?.predictionUrl;
    if (!predictionUrl) {
      errorText.textContent = "Set the prediction API URL in frontend/config.js, then try again.";
      setResultState("error");
      return;
    }

    setLoading(true);
    showMessage("");
    setResultState("empty");
    resultStatus.innerHTML = "<i></i> ANALYZING";
    resultStatus.classList.remove("is-ready", "is-error");

    try {
      const body = new FormData();
      body.append("image", selectedImage);
      const response = await fetch(predictionUrl, {
        method: "POST",
        body,
      });

      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error("The prediction service returned an unreadable response.");
      }
      if (!response.ok) {
        throw new Error(
          typeof data?.error === "string"
            ? data.error
            : `The prediction service returned an error (${response.status}).`,
        );
      }
      renderPrediction(normalizePrediction(data));
    } catch (error) {
      errorText.textContent =
        error instanceof TypeError
          ? "Could not reach the prediction API. Check that the backend is running and allows requests from this page."
          : error.message;
      setResultState("error");
    } finally {
      setLoading(false);
    }
  }

  input.addEventListener("change", () => acceptImage(input.files?.[0]));
  clearButton.addEventListener("click", clearImage);
  analyzeButton.addEventListener("click", analyzeImage);
  tryAgainButton.addEventListener("click", analyzeImage);
  addStockButton.addEventListener("click", addCurrentItemToStock);
  stockSearch.addEventListener("input", renderStockLog);
  stockRows.addEventListener("click", updateStockEntry);

  dropzone.addEventListener("keydown", (event) => {
    if (event.target === dropzone && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      input.click();
    }
  });
  for (const eventName of ["dragenter", "dragover"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "dragend"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (!dropzone.contains(event.relatedTarget)) {
        dropzone.classList.remove("is-dragging");
      }
    });
  }
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
    acceptImage(event.dataTransfer?.files?.[0]);
  });

  stockEntries = loadStockEntries();
  renderStockLog();
})();
