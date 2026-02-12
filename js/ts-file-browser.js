import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_ID = "ts.filebrowser";
const NODE_NAME = "TS_FileBrowser";
const ROUTE_BASE = "/ts_file_browser";

const folderSVG = `<svg viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg"><path d="M928 320H488L416 232c-15.1-18.9-38.3-29.9-63.1-29.9H128c-35.3 0-64 28.7-64 64v512c0 35.3 28.7 64 64 64h800c35.3 0 64-28.7 64-64V384c0-35.3-28.7-64-64-64z" fill="#F4D03F"></path></svg>`;
const videoSVG = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#FFFFFF" d="M3 5h11a2 2 0 0 1 2 2v1.5l4-2v11l-4-2V17a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z"/></svg>`;
const audioSVG = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#FFFFFF" d="M12 3v10.55A4 4 0 1 0 14 17V7h4V3h-6z"/></svg>`;

const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"];
const VIDEO_EXTS = [".mp4", ".webm", ".mov", ".mkv", ".avi"];
const AUDIO_EXTS = [".mp3", ".wav", ".ogg", ".flac", ".m4a"];

function normalizePath(path) {
    return (path || "").replace(/\\/g, "/");
}

function getMediaTypeByExt(filename) {
    const lower = (filename || "").toLowerCase();
    const dot = lower.lastIndexOf(".");
    const ext = dot >= 0 ? lower.slice(dot) : "";
    if (IMAGE_EXTS.includes(ext)) return "image";
    if (VIDEO_EXTS.includes(ext)) return "video";
    if (AUDIO_EXTS.includes(ext)) return "audio";
    return "file";
}

function createLazyLoader(rootEl) {
    if (!("IntersectionObserver" in window)) {
        return {
            observe(img, src) {
                img.src = src;
            },
        };
    }

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                const img = entry.target;
                const src = img.dataset.src;
                if (src) {
                    img.src = src;
                    delete img.dataset.src;
                }
                observer.unobserve(img);
            });
        },
        { root: rootEl, rootMargin: "200px" }
    );

    return {
        observe(img, src) {
            img.dataset.src = src;
            observer.observe(img);
        },
    };
}

function ensureLightbox() {
    const lightboxId = "tsfb-lightbox";
    let lightbox = document.getElementById(lightboxId);
    if (lightbox) return lightbox;

    lightbox = document.createElement("div");
    lightbox.id = lightboxId;
    lightbox.innerHTML = `
        <div class="tsfb-lightbox-inner">
            <button class="tsfb-lightbox-close">&times;</button>
            <div class="tsfb-lightbox-content">
                <img class="tsfb-lightbox-img" alt="Preview" />
                <video class="tsfb-lightbox-video" controls autoplay></video>
                <audio class="tsfb-lightbox-audio" controls autoplay></audio>
            </div>
        </div>
    `;
    document.body.appendChild(lightbox);

    lightbox.addEventListener("click", (e) => {
        if (e.target === lightbox) {
            lightbox.style.display = "none";
        }
    });
    lightbox.querySelector(".tsfb-lightbox-close").addEventListener("click", () => {
        lightbox.style.display = "none";
    });

    return lightbox;
}

function openLightbox(item) {
    const lightbox = ensureLightbox();
    const imgEl = lightbox.querySelector(".tsfb-lightbox-img");
    const videoEl = lightbox.querySelector(".tsfb-lightbox-video");
    const audioEl = lightbox.querySelector(".tsfb-lightbox-audio");

    imgEl.style.display = "none";
    videoEl.style.display = "none";
    audioEl.style.display = "none";
    videoEl.pause();
    audioEl.pause();

    if (item.type === "image") {
        imgEl.src = `${ROUTE_BASE}/view?filepath=${encodeURIComponent(item.path)}`;
        imgEl.style.display = "block";
    } else if (item.type === "video") {
        videoEl.src = `${ROUTE_BASE}/view?filepath=${encodeURIComponent(item.path)}`;
        videoEl.style.display = "block";
    } else if (item.type === "audio") {
        audioEl.src = `${ROUTE_BASE}/view?filepath=${encodeURIComponent(item.path)}`;
        audioEl.style.display = "block";
    }

    lightbox.style.display = "flex";
}

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            if (!this.properties) this.properties = {};
            if (!this.properties.gallery_unique_id) {
                this.properties.gallery_unique_id = `tsfb-${Math.random().toString(36).slice(2, 10)}`;
            }

            const node_instance = this;
            const uniqueId = `tsfb-${this.id}-${Math.random().toString(36).slice(2, 7)}`;

            const galleryIdWidget = this.addWidget(
                "hidden_text",
                "gallery_unique_id_widget",
                this.properties.gallery_unique_id,
                () => {},
                {}
            );
            galleryIdWidget.serializeValue = () => node_instance.properties.gallery_unique_id;
            galleryIdWidget.draw = function () {};
            galleryIdWidget.computeSize = () => [0, 0];

            const selectionWidget = this.addWidget(
                "hidden_text",
                "selection",
                this.properties.selection || "[]",
                () => {},
                { multiline: true }
            );
            selectionWidget.serializeValue = () => node_instance.properties.selection || "[]";
            selectionWidget.draw = function () {};
            selectionWidget.computeSize = () => [0, 0];

            const currentPathWidget = this.addWidget(
                "hidden_text",
                "current_path",
                this.properties.current_path || "",
                () => {},
                {}
            );
            currentPathWidget.serializeValue = () => node_instance.properties.current_path || "";
            currentPathWidget.draw = function () {};
            currentPathWidget.computeSize = () => [0, 0];

            const container = document.createElement("div");
            container.id = uniqueId;
            container.className = "tsfb-container";
            container.innerHTML = `
                <style>
                    #${uniqueId} { padding: 6px; color: #ddd; font-family: sans-serif; display: flex; flex-direction: column; box-sizing: border-box; overflow: hidden; position: relative; min-height: 0; }
                    #${uniqueId} .tsfb-controls { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
                    #${uniqueId} .tsfb-controls button, #${uniqueId} .tsfb-controls select { background: #333; color: #ddd; border: 1px solid #555; border-radius: 4px; padding: 4px 8px; }
                    #${uniqueId} .tsfb-controls button:disabled { opacity: 0.5; cursor: not-allowed; }
                    #${uniqueId} .tsfb-breadcrumb { display: flex; gap: 4px; flex-wrap: nowrap; overflow: hidden; }
                    #${uniqueId} .tsfb-breadcrumb span { cursor: pointer; white-space: nowrap; }
                    #${uniqueId} .tsfb-cardholder { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; overflow-y: auto; overflow-x: hidden; border: 1px solid #333; border-radius: 6px; padding: 6px; background: #1e1e1e; flex: 1 1 auto; min-height: 0; height: 0; box-sizing: border-box; width: 100%; }
                    #${uniqueId} .tsfb-card { background: #2a2a2a; border: 2px solid transparent; border-radius: 6px; padding: 6px; cursor: pointer; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; }
                    #${uniqueId} .tsfb-card.selected { border-color: #00FFC9; }
                    #${uniqueId} .tsfb-media { width: 100%; aspect-ratio: var(--tsfb-aspect, 1 / 1); display: flex; align-items: center; justify-content: center; background: #151515; border-radius: 4px; overflow: hidden; }
                    #${uniqueId} .tsfb-card img { width: 100%; height: 100%; object-fit: contain; display: block; }
                    #${uniqueId} .tsfb-card .tsfb-icon { width: 48px; height: 48px; }
                    #${uniqueId} .tsfb-card .tsfb-name { font-size: 11px; text-align: center; word-break: break-all; }
                    #${uniqueId} .tsfb-placeholder { color: #777; padding: 4px; position: absolute; left: 10px; top: 90px; z-index: 2; pointer-events: none; }
                    #tsfb-lightbox { position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: none; align-items: center; justify-content: center; z-index: 10000; }
                    #tsfb-lightbox .tsfb-lightbox-inner { position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }
                    #tsfb-lightbox .tsfb-lightbox-content { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }
                    #tsfb-lightbox img, #tsfb-lightbox video { max-width: 95%; max-height: 95%; object-fit: contain; }
                    #tsfb-lightbox audio { width: 70%; max-width: 600px; }
                    #tsfb-lightbox .tsfb-lightbox-close { position: absolute; top: 10px; right: 20px; width: 36px; height: 36px; border-radius: 50%; border: 2px solid #fff; background: rgba(0,0,0,0.5); color: #fff; font-size: 22px; cursor: pointer; }
                </style>
                <div class="tsfb-controls">
                    <button class="tsfb-choose">Select File</button>
                    <div class="tsfb-breadcrumb"></div>
                    <button class="tsfb-refresh">Refresh</button>
                    <button class="tsfb-delete" disabled>Delete</button>
                </div>
                <div class="tsfb-controls">
                    <label>Sort by:</label>
                    <select class="tsfb-sort-by">
                        <option value="name">Name</option>
                        <option value="date">Date</option>
                        <option value="type">Type</option>
                    </select>
                    <label>Order:</label>
                    <select class="tsfb-sort-order">
                        <option value="asc">Ascending</option>
                        <option value="desc">Descending</option>
                    </select>
                    <label>Images</label><input type="checkbox" class="tsfb-show-images" checked>
                    <label>Videos</label><input type="checkbox" class="tsfb-show-videos" checked>
                    <label>Audio</label><input type="checkbox" class="tsfb-show-audio" checked>
                </div>
                <div class="tsfb-cardholder"></div>
                <div class="tsfb-placeholder">Loading...</div>
            `;

            this.addDOMWidget("ts_file_browser", "div", container, {});
            container.style.height = "100%";
            container.style.width = "100%";
            this.size = [820, 640];

            const cardholder = container.querySelector(".tsfb-cardholder");
            const placeholder = container.querySelector(".tsfb-placeholder");
            const breadcrumbEl = container.querySelector(".tsfb-breadcrumb");
            const chooseButton = container.querySelector(".tsfb-choose");
            const refreshButton = container.querySelector(".tsfb-refresh");
            const deleteButton = container.querySelector(".tsfb-delete");
            const sortBySelect = container.querySelector(".tsfb-sort-by");
            const sortOrderSelect = container.querySelector(".tsfb-sort-order");
            const showImagesCheckbox = container.querySelector(".tsfb-show-images");
            const showVideosCheckbox = container.querySelector(".tsfb-show-videos");
            const showAudioCheckbox = container.querySelector(".tsfb-show-audio");
            const lazyLoader = createLazyLoader(cardholder);
            const filePicker = document.createElement("input");
            filePicker.type = "file";
            filePicker.accept = [...IMAGE_EXTS, ...VIDEO_EXTS, ...AUDIO_EXTS].join(",");
            filePicker.style.display = "none";
            container.appendChild(filePicker);

            let rootDirectory = "";
            let currentDirectory = "";
            let inputRoot = "";
            let selection = [];
            let currentPage = 1;
            let totalPages = 1;
            let isLoading = false;
            let parentDir = null;
            let sizeRaf = null;
            const scheduleSizeSync = () => {
                if (sizeRaf) return;
                sizeRaf = requestAnimationFrame(() => {
                    sizeRaf = null;
                    if (this.onResize) {
                        this.onResize(this.size);
                    }
                });
            };

            const updateSelectionState = () => {
                const selectionJson = JSON.stringify(selection);
                this.setProperty("selection", selectionJson);
                const widget = this.widgets.find(w => w.name === "selection");
                if (widget) widget.value = selectionJson;
                deleteButton.disabled = selection.length === 0;
            };

            const setCurrentDirectory = (path, saveState = false) => {
                const normalized = normalizePath(path || rootDirectory);
                currentDirectory = normalized || rootDirectory;
                this.properties.current_path = currentDirectory;
                if (currentPathWidget) currentPathWidget.value = currentDirectory;
                renderBreadcrumb(currentDirectory);
                if (saveState) saveCurrentControlsState();
            };

            const renderBreadcrumb = (path) => {
                breadcrumbEl.innerHTML = "";
                const parts = normalizePath(path).split("/").filter(Boolean);
                let builtPath = "";
                if (path.match(/^[a-zA-Z]:\//)) {
                    const drive = parts.shift();
                    builtPath = `${drive}/`;
                    const driveEl = document.createElement("span");
                    driveEl.textContent = drive;
                    driveEl.dataset.path = builtPath;
                    breadcrumbEl.appendChild(driveEl);
                } else {
                    builtPath = "/";
                    const rootEl = document.createElement("span");
                    rootEl.textContent = "/";
                    rootEl.dataset.path = builtPath;
                    breadcrumbEl.appendChild(rootEl);
                }

                parts.forEach(part => {
                    const sep = document.createElement("span");
                    sep.textContent = ">";
                    sep.style.color = "#666";
                    sep.style.margin = "0 4px";
                    breadcrumbEl.appendChild(sep);

                    builtPath += `${part}/`;
                    const el = document.createElement("span");
                    el.textContent = part;
                    el.dataset.path = builtPath;
                    breadcrumbEl.appendChild(el);
                });
            };

            const saveCurrentControlsState = () => {
                const state = {
                    sort_by: sortBySelect.value,
                    sort_order: sortOrderSelect.value,
                    show_images: showImagesCheckbox.checked,
                    show_videos: showVideosCheckbox.checked,
                    show_audio: showAudioCheckbox.checked,
                    current_path: currentDirectory,
                    selection: selection,
                };
                api.fetchApi(`${ROUTE_BASE}/set_ui_state`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ node_id: this.id, gallery_id: this.properties.gallery_unique_id, state }),
                }).catch(() => {});
            };

            const createCard = (item) => {
                const card = document.createElement("div");
                card.className = "tsfb-card";
                card.dataset.path = item.path;
                card.dataset.type = item.type;
                card.title = item.name;

                if (item.type === "dir") {
                    const icon = document.createElement("div");
                    icon.className = "tsfb-icon";
                    icon.innerHTML = folderSVG;
                    const name = document.createElement("div");
                    name.className = "tsfb-name";
                    name.textContent = item.name;
                    card.appendChild(icon);
                    card.appendChild(name);
                } else if (item.type === "image" || item.type === "video") {
                    const media = document.createElement("div");
                    media.className = "tsfb-media";
                    const img = document.createElement("img");
                    img.alt = item.name;
                    img.dataset.fallback = "thumb";
                    const thumbUrl = `${ROUTE_BASE}/thumbnail?filepath=${encodeURIComponent(item.path)}&t=${item.mtime}`;
                    lazyLoader.observe(img, thumbUrl);

                    img.addEventListener("load", () => {
                        if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                            media.style.setProperty("--tsfb-aspect", `${img.naturalWidth} / ${img.naturalHeight}`);
                        }
                    });

                    img.addEventListener("error", () => {
                        if (item.type === "image" && img.dataset.fallback === "thumb") {
                            img.dataset.fallback = "view";
                            img.src = `${ROUTE_BASE}/view?filepath=${encodeURIComponent(item.path)}&t=${item.mtime}`;
                            return;
                        }
                        const icon = document.createElement("div");
                        icon.className = "tsfb-icon";
                        icon.innerHTML = item.type === "video" ? videoSVG : folderSVG;
                        const name = document.createElement("div");
                        name.className = "tsfb-name";
                        name.textContent = item.name;
                        card.innerHTML = "";
                        card.appendChild(icon);
                        card.appendChild(name);
                    }, { once: false });

                    const name = document.createElement("div");
                    name.className = "tsfb-name";
                    name.textContent = item.name;
                    media.appendChild(img);
                    card.appendChild(media);
                    card.appendChild(name);
                } else if (item.type === "audio") {
                    const icon = document.createElement("div");
                    icon.className = "tsfb-icon";
                    icon.innerHTML = audioSVG;
                    const name = document.createElement("div");
                    name.className = "tsfb-name";
                    name.textContent = item.name;
                    card.appendChild(icon);
                    card.appendChild(name);
                }

                if (selection.some(s => s.path === item.path)) {
                    card.classList.add("selected");
                }
                return card;
            };

            const renderItems = (items, append = false) => {
                if (!append) cardholder.innerHTML = "";
                items.forEach(item => {
                    const card = createCard(item);
                    cardholder.appendChild(card);
                });
                scheduleSizeSync();
            };

            const fetchItems = async (page = 1, append = false, forceRefresh = false) => {
                if (isLoading) return;
                isLoading = true;
                if (!append) {
                    currentPage = 1;
                    placeholder.textContent = "Loading...";
                    placeholder.style.display = "block";
                }

                const directory = currentDirectory || rootDirectory;
                if (!directory) {
                    placeholder.textContent = "Input folder is not available.";
                    placeholder.style.display = "block";
                    isLoading = false;
                    return;
                }

                let url = `${ROUTE_BASE}/images?directory=${encodeURIComponent(directory)}&page=${page}&sort_by=${sortBySelect.value}&sort_order=${sortOrderSelect.value}&show_images=${showImagesCheckbox.checked}&show_videos=${showVideosCheckbox.checked}&show_audio=${showAudioCheckbox.checked}&force_refresh=${forceRefresh}`;
                if (selection.length > 0) {
                    selection.forEach(item => { url += `&selected_paths=${encodeURIComponent(item.path)}`; });
                }

                try {
                    const response = await api.fetchApi(url);
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.error || "Request failed");
                    }
                    const data = await response.json();
                    totalPages = data.total_pages || 1;
                    parentDir = data.parent_directory;
                    setCurrentDirectory(data.current_directory || directory, false);
                    const items = data.items || [];
                    if (!append) {
                        renderItems(items, false);
                        cardholder.scrollTop = 0;
                    } else {
                        renderItems(items, true);
                    }

                    placeholder.style.display = items.length === 0 && page === 1 ? "block" : "none";
                    placeholder.textContent = items.length === 0 ? "The folder is empty." : "";
                    currentPage = page;
                } catch (e) {
                    placeholder.textContent = `Error: ${e.message}`;
                    placeholder.style.display = "block";
                } finally {
                    isLoading = false;
                }
            };

            const resetAndReload = (forceRefresh = false) => {
                saveCurrentControlsState();
                fetchItems(1, false, forceRefresh);
            };

            breadcrumbEl.addEventListener("click", (e) => {
                const target = e.target.closest("span");
                if (!target || !target.dataset.path) return;
                setCurrentDirectory(target.dataset.path, true);
                resetAndReload(true);
            });

            chooseButton.addEventListener("click", () => {
                (async () => {
                    try {
                        const response = await api.fetchApi(`${ROUTE_BASE}/pick_file`);
                        if (response.ok) {
                            const data = await response.json();
                            if (data && data.path) {
                                const normalized = normalizePath(data.path);
                                const lastSlash = normalized.lastIndexOf("/");
                                const dir = lastSlash >= 0 ? normalized.slice(0, lastSlash + 1) : normalized;
                                const filename = lastSlash >= 0 ? normalized.slice(lastSlash + 1) : normalized;
                                const type = getMediaTypeByExt(filename);
                                rootDirectory = dir || rootDirectory;
                                setCurrentDirectory(dir || rootDirectory, true);
                                selection = filename ? [{ path: normalized, type }] : [];
                                updateSelectionState();
                                saveCurrentControlsState();
                                resetAndReload(true);
                                return;
                            }
                        }
                    } catch (e) {
                    }

                    filePicker.value = "";
                    filePicker.click();
                })();
            });

            filePicker.addEventListener("change", async () => {
                const file = filePicker.files?.[0];
                if (!file) return;

                const form = new FormData();
                form.append("image", file);
                form.append("type", "input");

                try {
                    const response = await api.fetchApi("/upload/image", {
                        method: "POST",
                        body: form,
                    });
                    if (!response.ok) {
                        throw new Error("Upload failed");
                    }
                    const data = await response.json();
                    const filename = data?.name || file.name;
                    const subfolder = data?.subfolder || "";
                    const baseRoot = inputRoot || rootDirectory;
                    const selectedDir = normalizePath(subfolder ? `${baseRoot}/${subfolder}` : baseRoot);
                    const selectedPath = normalizePath(subfolder ? `${selectedDir}/${filename}` : `${baseRoot}/${filename}`);
                    const type = getMediaTypeByExt(filename);

                    setCurrentDirectory(selectedDir, true);
                    selection = [{ path: selectedPath, type }];
                    updateSelectionState();
                    saveCurrentControlsState();
                    resetAndReload(true);
                } catch (e) {
                    alert(`Upload failed: ${e}`);
                }
            });

            refreshButton.addEventListener("click", () => resetAndReload(true));
            sortBySelect.addEventListener("change", () => resetAndReload(false));
            sortOrderSelect.addEventListener("change", () => resetAndReload(false));
            showImagesCheckbox.addEventListener("change", () => resetAndReload(false));
            showVideosCheckbox.addEventListener("change", () => resetAndReload(false));
            showAudioCheckbox.addEventListener("change", () => resetAndReload(false));

            deleteButton.addEventListener("click", async () => {
                if (selection.length === 0) return;
                const filepaths = selection.map(s => s.path);
                if (!confirm(`Delete ${selection.length} selected file(s)?`)) return;
                try {
                    await api.fetchApi(`${ROUTE_BASE}/delete_files`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ filepaths }),
                    });
                    selection = [];
                    updateSelectionState();
                    resetAndReload(true);
                } catch (e) {
                    alert(`Delete failed: ${e}`);
                }
            });

            cardholder.addEventListener("click", (event) => {
                const card = event.target.closest(".tsfb-card");
                if (!card) return;
                const type = card.dataset.type;
                const path = card.dataset.path;

                if (type === "dir") {
                    setCurrentDirectory(path, true);
                    resetAndReload(true);
                    return;
                }

                const index = selection.findIndex(item => item.path === path);
                const isMulti = event.ctrlKey || event.metaKey;
                if (isMulti) {
                    if (index > -1) {
                        selection.splice(index, 1);
                        card.classList.remove("selected");
                    } else {
                        selection.push({ path, type });
                        card.classList.add("selected");
                    }
                } else {
                    cardholder.querySelectorAll(".tsfb-card.selected").forEach(c => c.classList.remove("selected"));
                    if (index > -1 && selection.length === 1) {
                        selection = [];
                        card.classList.remove("selected");
                    } else {
                        selection = [{ path, type }];
                        card.classList.add("selected");
                    }
                }

                updateSelectionState();
                saveCurrentControlsState();
            });

            cardholder.addEventListener("dblclick", (event) => {
                const card = event.target.closest(".tsfb-card");
                if (!card) return;
                const type = card.dataset.type;
                if (!["image", "video", "audio"].includes(type)) return;
                openLightbox({ path: card.dataset.path, type });
            });

            cardholder.addEventListener("scroll", () => {
                if (cardholder.scrollTop + cardholder.clientHeight >= cardholder.scrollHeight - 200) {
                    if (!isLoading && currentPage < totalPages) {
                        fetchItems(currentPage + 1, true, false);
                    }
                }
            });

            this.onResize = function (size) {
                const minHeight = 470;
                const minWidth = 800;
                if (size[1] < minHeight) size[1] = minHeight;
                if (size[0] < minWidth) size[0] = minWidth;
                if (!container) return;
                let topOffset = container.offsetTop;
                const approximateHeaderHeight = 120;
                if (topOffset < 20) {
                    topOffset += approximateHeaderHeight;
                }
                const bottomPadding = 20;
                const targetHeight = size[1] - topOffset - bottomPadding;
                container.style.height = `${Math.max(200, targetHeight)}px`;
                container.style.width = "100%";
            };

            const initializeNode = async () => {
                try {
                    const galleryId = this.properties.gallery_unique_id;
                    const response = await api.fetchApi(`${ROUTE_BASE}/get_ui_state?node_id=${this.id}&gallery_id=${galleryId}`);
                    const state = await response.json();
                    if (state) {
                        sortBySelect.value = state.sort_by || "name";
                        sortOrderSelect.value = state.sort_order || "asc";
                        showImagesCheckbox.checked = state.show_images !== false;
                        showVideosCheckbox.checked = state.show_videos !== false;
                        showAudioCheckbox.checked = state.show_audio !== false;
                        selection = state.selection || [];
                        updateSelectionState();

                        inputRoot = state.input_root || inputRoot;
                        rootDirectory = state.current_path || inputRoot || rootDirectory;
                        const initialPath = state.current_path || rootDirectory;
                        if (rootDirectory) setCurrentDirectory(initialPath || rootDirectory, false);

                        fetchItems(1, false, false);
                    }
                } catch (e) {
                    console.error("[TS File Browser] init error:", e);
                    fetchItems(1, false, false);
                }
            };

            requestAnimationFrame(() => {
                if (this.onResize) this.onResize(this.size);
            });

            setTimeout(() => initializeNode.call(this), 1);

            const onRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (sizeRaf) cancelAnimationFrame(sizeRaf);
                return onRemoved?.apply(this, arguments);
            };

            return r;
        };
    },
});
