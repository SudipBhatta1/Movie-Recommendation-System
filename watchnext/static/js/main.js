// ── Watchlist (localStorage) ──────────────────────────────────────────────────
function getWatchlist() {
  try { return JSON.parse(localStorage.getItem('watchlist') || '[]'); } catch { return []; }
}
function saveWatchlist(wl) {
  localStorage.setItem('watchlist', JSON.stringify(wl));
}
function toggleWatchlist(id, title, poster) {
  let wl = getWatchlist();
  const idx = wl.findIndex(w => w.id == id);
  if (idx >= 0) { wl.splice(idx, 1); showToast(`Removed "${title}" from watchlist`); }
  else { wl.push({ id, title, poster }); showToast(`Added "${title}" to watchlist`); }
  saveWatchlist(wl);
  updateWlBadge();
  syncWatchlistButtons();
  renderWatchlistDrawer();
}
function syncWatchlistButtons() {
  const wl = getWatchlist();
  document.querySelectorAll('.wl-btn').forEach(btn => {
    const id = btn.id?.replace('wl-','');
    if (id) btn.classList.toggle('active', wl.some(w => w.id == id));
  });
}
function updateWlBadge() {
  const count = getWatchlist().length;
  const badge = document.getElementById('navWlBadge');
  if (!badge) return;
  badge.textContent = count;
  badge.style.display = count > 0 ? 'inline-flex' : 'none';
}
function openWatchlist() {
  renderWatchlistDrawer();
  document.getElementById('drawerOverlay').classList.add('active');
}
function closeWatchlist() {
  document.getElementById('drawerOverlay').classList.remove('active');
}
function renderWatchlistDrawer() {
  const wl = getWatchlist();
  const body = document.getElementById('watchlistBody');
  if (!body) return;
  if (!wl.length) {
    body.innerHTML = '<p class="empty-msg">Your watchlist is empty.<br>Click 🔖 on any movie!</p>';
    return;
  }
  body.innerHTML = wl.map(w => `
    <div class="wl-item">
      <img src="${w.poster}" alt="${w.title}" onerror="this.style.display='none'">
      <div class="wl-item-info">
        <strong>${w.title}</strong>
        <a href="/movie/${w.id}">View →</a>
      </div>
      <button class="wl-remove" onclick="toggleWatchlist(${w.id},'${w.title.replace(/'/g,"\\'")}','${w.poster.replace(/'/g,"\\'")}')">✕</button>
    </div>
  `).join('');
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg) {
  let t = document.getElementById('toast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'toast';
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), 2800);
}

// ── Flash auto-dismiss ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach(f => {
    setTimeout(() => { f.style.opacity='0'; setTimeout(()=>f.remove(),400); }, 4000);
  });
  updateWlBadge();
  syncWatchlistButtons();
});

function togglePassword() {
    const input = document.getElementById('passwordInput');
    const btn = document.querySelector('.eye-btn');
    if (input.type === 'password') {
        input.type = 'text';
        btn.textContent = '🙈';
    } else {
        input.type = 'password';
        btn.textContent = '👁';
    }
}