(() => {
  const CART_KEY = "threadline-demo-bag-v1";
  const grid = document.querySelector("#product-grid");
  const search = document.querySelector("#product-search");
  const filters = document.querySelector("#shop-categories");
  const countLabel = document.querySelector("#collection-count");
  const emptyState = document.querySelector("#shop-empty");
  const noResults = document.querySelector("#shop-no-results");
  const bagButton = document.querySelector("#bag-button");
  const bagDrawer = document.querySelector("#bag-drawer");
  const bagBackdrop = document.querySelector("#bag-backdrop");
  const bagItems = document.querySelector("#bag-items");
  const bagEmpty = document.querySelector("#bag-empty");
  const bagSummary = document.querySelector("#bag-summary");
  const bagCount = document.querySelector("#bag-count");
  const bagTitleCount = document.querySelector("#bag-title-count");
  const bagTotalItems = document.querySelector("#bag-total-items");
  const toast = document.querySelector("#shop-toast");
  const currency = new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  });

  let entries = [];
  let cart = loadCart();
  let activeCategory = "All pieces";
  let toastTimeout;
  const imageUrls = new Map();

  function loadCart() {
    try {
      const saved = JSON.parse(localStorage.getItem(CART_KEY) ?? "[]");
      if (!Array.isArray(saved)) return [];
      return saved.filter(
        (item) =>
          item &&
          typeof item.id === "string" &&
          Number.isInteger(item.quantity) &&
          item.quantity > 0,
      );
    } catch (error) {
      showToast(`Your saved bag could not be read: ${error.message}`);
      return [];
    }
  }

  function saveCart() {
    try {
      localStorage.setItem(CART_KEY, JSON.stringify(cart));
      return true;
    } catch (error) {
      showToast(`Your bag could not be saved in this browser: ${error.message}`);
      return false;
    }
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add("is-visible");
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => toast.classList.remove("is-visible"), 2800);
  }

  function availableProducts() {
    return entries.filter(
      (entry) => entry.units > 0 && (!entry.needsReview || entry.reviewed),
    );
  }

  function priceText(price) {
    return typeof price === "number" && Number.isFinite(price) && price >= 0
      ? currency.format(price)
      : "Ask in boutique";
  }

  function releaseImageUrls(prefix) {
    for (const [key, url] of imageUrls) {
      if (key.startsWith(prefix)) {
        URL.revokeObjectURL(url);
        imageUrls.delete(key);
      }
    }
  }

  function makePlaceholder(entry) {
    const placeholder = document.createElement("div");
    placeholder.className = "product-placeholder";
    const initial = document.createElement("span");
    initial.className = "placeholder-mark";
    initial.textContent = entry.productName?.trim().charAt(0).toLocaleUpperCase() || "T";
    const category = document.createElement("span");
    category.className = "placeholder-category";
    category.textContent = entry.label;
    placeholder.append(initial, category);
    return placeholder;
  }

  async function attachProductImage(imageFrame, entry, imageKey) {
    if (!entry.imageId) return;
    try {
      const image = await window.ThreadlineCatalog.getImage(entry.imageId);
      if (!image || !imageFrame.isConnected) return;
      const url = URL.createObjectURL(image);
      imageUrls.set(imageKey, url);
      const img = document.createElement("img");
      img.src = url;
      img.alt = entry.productName || entry.label;
      img.loading = "lazy";
      imageFrame.replaceChildren(img);
    } catch (error) {
      showToast(`A product photo could not be loaded: ${error.message}`);
    }
  }

  function makeProductCard(entry) {
    const card = document.createElement("article");
    card.className = "product-card";
    const imageFrame = document.createElement("div");
    imageFrame.className = "product-image-frame";
    imageFrame.append(makePlaceholder(entry));

    const tag = document.createElement("span");
    tag.className = "product-tag";
    tag.textContent = new Date(entry.createdAt).toDateString() === new Date().toDateString()
      ? "JUST IN"
      : "BOUTIQUE PICK";
    imageFrame.append(tag);
    const photo = document.createElement("span");
    photo.className = "product-photo-note";
    photo.textContent = entry.imageId ? "SHOP PHOTO" : "READY TO DISCOVER";
    imageFrame.append(photo);
    attachProductImage(imageFrame, entry, `product:${entry.id}`);

    const details = document.createElement("div");
    details.className = "product-details";
    const titleLine = document.createElement("div");
    titleLine.className = "product-title-line";
    const title = document.createElement("h3");
    title.textContent = entry.productName || entry.label;
    const price = document.createElement("strong");
    price.textContent = priceText(entry.price);
    titleLine.append(title, price);

    const category = document.createElement("p");
    category.className = "product-category";
    category.textContent = entry.label;
    const meta = document.createElement("div");
    meta.className = "product-meta";
    const variant = document.createElement("span");
    variant.textContent = entry.variant || "One lovely find";
    const availability = document.createElement("span");
    availability.textContent = entry.units <= 3 ? `${entry.units} left` : "In the boutique";
    meta.append(variant, availability);

    const addButton = document.createElement("button");
    addButton.type = "button";
    addButton.className = "product-add-button";
    addButton.dataset.productId = entry.id;
    addButton.textContent = "Add to bag";
    details.append(titleLine, category, meta, addButton);
    card.append(imageFrame, details);
    return card;
  }

  function renderProducts() {
    releaseImageUrls("product:");
    const products = availableProducts();
    const term = search.value.trim().toLocaleLowerCase();
    const visible = products.filter((entry) => {
      const matchesCategory =
        activeCategory === "All pieces" || entry.label === activeCategory;
      const searchable = `${entry.productName} ${entry.label} ${entry.variant}`.toLocaleLowerCase();
      return matchesCategory && searchable.includes(term);
    });
    grid.replaceChildren(...visible.map(makeProductCard));
    countLabel.textContent = `${products.length} ${products.length === 1 ? "PIECE" : "PIECES"}`;
    emptyState.hidden = products.length > 0;
    noResults.hidden = products.length === 0 || visible.length > 0;
    renderCategoryFilters(products);
  }

  function renderCategoryFilters(products) {
    const categories = ["All pieces", ...new Set(products.map((entry) => entry.label))];
    if (!categories.includes(activeCategory)) activeCategory = "All pieces";
    filters.replaceChildren(
      ...categories.map((category) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `category-filter${category === activeCategory ? " is-active" : ""}`;
        button.setAttribute("aria-pressed", String(category === activeCategory));
        button.textContent = category;
        button.addEventListener("click", () => {
          activeCategory = category;
          renderProducts();
        });
        return button;
      }),
    );
  }

  function cartEntries() {
    const products = new Map(availableProducts().map((entry) => [entry.id, entry]));
    cart = cart
      .filter((item) => products.has(item.id))
      .map((item) => ({
        ...item,
        quantity: Math.min(item.quantity, products.get(item.id).units),
      }))
      .filter((item) => item.quantity > 0);
    return cart
      .map((item) => ({ ...products.get(item.id), quantity: item.quantity }))
      .filter(Boolean);
  }

  function renderBag() {
    const items = cartEntries();
    saveCart();
    const total = items.reduce((sum, item) => sum + item.quantity, 0);
    bagCount.textContent = total > 99 ? "99+" : String(total);
    bagTitleCount.textContent = `(${total})`;
    bagTotalItems.textContent = String(total);
    bagEmpty.hidden = items.length > 0;
    bagSummary.hidden = items.length === 0;
    bagItems.replaceChildren();
    releaseImageUrls("bag:");

    for (const item of items) {
      const row = document.createElement("article");
      row.className = "bag-item";
      const thumb = document.createElement("div");
      thumb.className = "bag-item-thumb";
      thumb.append(makePlaceholder(item));
      attachProductImage(thumb, item, `bag:${item.id}`);

      const info = document.createElement("div");
      info.className = "bag-item-info";
      const name = document.createElement("strong");
      name.textContent = item.productName || item.label;
      const category = document.createElement("span");
      category.textContent = item.variant ? `${item.label} · ${item.variant}` : item.label;
      const price = document.createElement("span");
      price.textContent = priceText(item.price);
      const quantity = document.createElement("div");
      quantity.className = "bag-quantity";
      const minus = document.createElement("button");
      minus.type = "button";
      minus.dataset.cartAction = "decrease";
      minus.dataset.productId = item.id;
      minus.setAttribute("aria-label", `Remove one ${item.productName}`);
      minus.textContent = "−";
      const value = document.createElement("span");
      value.textContent = String(item.quantity);
      const plus = document.createElement("button");
      plus.type = "button";
      plus.dataset.cartAction = "increase";
      plus.dataset.productId = item.id;
      plus.disabled = item.quantity >= item.units;
      plus.setAttribute("aria-label", `Add one ${item.productName}`);
      plus.textContent = "+";
      quantity.append(minus, value, plus);
      info.append(name, category, price, quantity);

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "bag-item-remove";
      remove.dataset.cartAction = "remove";
      remove.dataset.productId = item.id;
      remove.textContent = "Remove";
      row.append(thumb, info, remove);
      bagItems.append(row);
    }
  }

  function addToBag(productId) {
    const product = availableProducts().find((entry) => entry.id === productId);
    if (!product) {
      showToast("This piece is no longer available in the boutique.");
      return;
    }
    const existing = cart.find((item) => item.id === productId);
    if (existing) {
      if (existing.quantity >= product.units) {
        showToast("That is all the available stock for this piece.");
        return;
      }
      existing.quantity += 1;
    } else {
      cart.push({ id: productId, quantity: 1 });
    }
    if (!saveCart()) return;
    renderBag();
    showToast(`${product.productName || product.label} added to your bag.`);
  }

  function updateBag(event) {
    const button = event.target.closest("button[data-cart-action]");
    if (!button) return;
    const item = cart.find((entry) => entry.id === button.dataset.productId);
    if (!item) return;
    if (button.dataset.cartAction === "remove") {
      cart = cart.filter((entry) => entry.id !== item.id);
    } else if (button.dataset.cartAction === "decrease") {
      item.quantity -= 1;
      if (item.quantity === 0) cart = cart.filter((entry) => entry.id !== item.id);
    } else if (button.dataset.cartAction === "increase") {
      const product = availableProducts().find((entry) => entry.id === item.id);
      if (!product || item.quantity >= product.units) return;
      item.quantity += 1;
    }
    saveCart();
    renderBag();
  }

  function setBagOpen(open) {
    bagDrawer.classList.toggle("is-open", open);
    bagDrawer.setAttribute("aria-hidden", String(!open));
    bagBackdrop.hidden = !open;
    document.body.classList.toggle("bag-is-open", open);
    if (open) document.querySelector("#bag-close").focus();
    else bagButton.focus();
  }

  try {
    entries = window.ThreadlineCatalog.loadEntries();
  } catch (error) {
    showToast(`The boutique catalog could not be loaded: ${error.message}`);
    entries = [];
  }
  renderProducts();
  renderBag();

  search.addEventListener("input", renderProducts);
  grid.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-product-id]");
    if (button) addToBag(button.dataset.productId);
  });
  bagButton.addEventListener("click", () => {
    renderBag();
    setBagOpen(true);
  });
  document.querySelector("#bag-close").addEventListener("click", () => setBagOpen(false));
  document.querySelector("#continue-shopping").addEventListener("click", () => setBagOpen(false));
  bagBackdrop.addEventListener("click", () => setBagOpen(false));
  bagItems.addEventListener("click", updateBag);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && bagDrawer.classList.contains("is-open")) setBagOpen(false);
  });
})();
