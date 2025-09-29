document.addEventListener('DOMContentLoaded', () => {
    const tg = window.Telegram.WebApp;

    const outputEl = document.getElementById('output');
    const readinessMessage = document.getElementById('readiness-message');
    const unsupportedMessage = document.getElementById('unsupported-message');

    let cloudStorage, sessionStorage;
    let isTelegramEnv = false;

    // --- Initialization ---
    try {
        if (tg.initData) {
            tg.ready();
            tg.expand();
            isTelegramEnv = true;
            cloudStorage = tg.CloudStorage;
            sessionStorage = tg.SessionStorage;
            readinessMessage.classList.remove('hidden');
        } else {
            unsupportedMessage.classList.remove('hidden');
            // Fallback to browser storage for local testing
            cloudStorage = createLocalStoragePolyfill('cloud_');
            sessionStorage = createLocalStoragePolyfill('session_');
        }
    } catch (error) {
        console.error("Telegram WebApp environment not found.", error);
        unsupportedMessage.classList.remove('hidden');
        cloudStorage = createLocalStoragePolyfill('cloud_');
        sessionStorage = createLocalStoragePolyfill('session_');
    }

    // --- UI Elements ---
    // Cloud
    const cloudKeyInput = document.getElementById('cloud-key');
    const cloudValueInput = document.getElementById('cloud-value');
    const cloudSaveBtn = document.getElementById('cloud-save');
    const cloudGetBtn = document.getElementById('cloud-get');
    const cloudDeleteBtn = document.getElementById('cloud-delete');
    // Session
    const sessionKeyInput = document.getElementById('session-key');
    const sessionValueInput = document.getElementById('session-value');
    const sessionSaveBtn = document.getElementById('session-save');
    const sessionGetBtn = document.getElementById('session-get');
    const sessionDeleteBtn = document.getElementById('session-delete');

    // --- Event Listeners ---
    cloudSaveBtn.addEventListener('click', () => handleSet(cloudStorage, cloudKeyInput.value, cloudValueInput.value));
    cloudGetBtn.addEventListener('click', () => handleGet(cloudStorage, cloudKeyInput.value));
    cloudDeleteBtn.addEventListener('click', () => handleDelete(cloudStorage, cloudKeyInput.value));

    sessionSaveBtn.addEventListener('click', () => handleSet(sessionStorage, sessionKeyInput.value, sessionValueInput.value));
    sessionGetBtn.addEventListener('click', () => handleGet(sessionStorage, sessionKeyInput.value));
    sessionDeleteBtn.addEventListener('click', () => handleDelete(sessionStorage, sessionKeyInput.value));

    // --- Handlers ---
    function handleSet(storage, key, value) {
        if (!key) {
            displayOutput('Error: Key cannot be empty.');
            return;
        }
        storage.setItem(key, value, (err, success) => {
            if (err) {
                displayOutput(`Error setting item: ${err}`);
            } else if (success) {
                displayOutput(`Successfully saved: { "${key}": "${value}" }`);
            } else {
                 displayOutput(`Operation completed. { "${key}": "${value}" }`);
            }
        });
    }

    function handleGet(storage, key) {
        if (!key) {
            displayOutput('Error: Key cannot be empty.');
            return;
        }
        storage.getItem(key, (err, value) => {
            if (err) {
                displayOutput(`Error getting item: ${err}`);
            } else if (value === null || value === undefined) {
                 displayOutput(`No value found for key: "${key}"`);
            }
            else {
                displayOutput(`Value for "${key}":\n${value}`);
            }
        });
    }

    function handleDelete(storage, key) {
        if (!key) {
            displayOutput('Error: Key cannot be empty.');
            return;
        }
        storage.removeItem(key, (err, success) => {
             if (err) {
                displayOutput(`Error deleting item: ${err}`);
            } else if (success) {
                displayOutput(`Successfully deleted key: "${key}"`);
            } else {
                 displayOutput(`Key "${key}" not found or already deleted.`);
            }
        });
    }

    function displayOutput(message) {
        outputEl.textContent = JSON.stringify(message, null, 2).replace(/^"|"$/g, ''); // Pretty print
    }

    // --- Polyfill for local browser testing ---
    function createLocalStoragePolyfill(prefix) {
        return {
            setItem: (key, value, callback) => {
                try {
                    localStorage.setItem(prefix + key, value);
                    if (callback) callback(null, true);
                } catch (e) {
                    if (callback) callback(e, false);
                }
            },
            getItem: (key, callback) => {
                try {
                    const value = localStorage.getItem(prefix + key);
                    if (callback) callback(null, value);
                } catch (e) {
                    if (callback) callback(e, null);
                }
            },
            removeItem: (key, callback) => {
                try {
                    localStorage.removeItem(prefix + key);
                    if (callback) callback(null, true);
                } catch (e) {
                    if (callback) callback(e, false);
                }
            }
        };
    }
});