/** main.js ─ 공통 스크립트 */
console.info('Tech Blog loaded:', new Date().toISOString().slice(0, 10));

document.addEventListener('DOMContentLoaded', () => {
  /* ───────────────────────── 1. 카테고리 Chip 필터 (index) ───────────────────────── */
  const chips  = document.querySelectorAll('.chip');
  const cards  = document.querySelectorAll('.card');

  if (chips.length) {
    chips.forEach(chip =>
      chip.addEventListener('click', () => {
        chips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');

        const f = chip.dataset.filter;
        cards.forEach(card => {
          card.style.display = (f === 'all' || card.dataset.cat === f) ? '' : 'none';
        });
      })
    );
  }

  /* ───────────────────────── 2. 글 페이지에 Topbar · Footer 동적 주입 ───────────────────────── */
  /* index.html(홈)은 이미 들어있으므로 .topbar 유무로 판단 */
  if (!document.querySelector('.topbar')) {
    /* 2-A) Topbar */
    const bar = document.createElement('header');
    bar.className = 'topbar';
    bar.innerHTML = `
      <a class="logo" href="/">Sangmin&nbsp;Lee</a>
      <nav class="top-nav">
        <a href="/">Bio</a>
        <a href="/about/">About</a>
      </nav>`;
    document.body.prepend(bar);
    document.body.style.paddingTop = '48px';      // 글 상단 안 잘리도록
  }

  if (!document.querySelector('.site-footer')) {
    /* 2-B) Footer */
    const foot = document.createElement('footer');
    foot.className = 'site-footer';
    foot.innerHTML = `
      <div class="footer-copy">© 2025 Sangmin Lee</div>
          <ul class="social">
      <li><a href="https://linkedin.com/in/your-id" aria-label="LinkedIn">
            <i class="fab fa-linkedin"></i></a></li>
      <li><a href="https://github.com/san9min" aria-label="GitHub">
            <i class="fab fa-github"></i></a></li>
    </ul>`;
    document.body.append(foot);
  }
});

/* ───────────────────────── 3. 스크롤 32 px ↑ → Topbar 불투명 ───────────────────────── */
window.addEventListener('scroll', () => {
  const tb = document.querySelector('.topbar');
  if (tb) tb.classList.toggle('scrolled', window.scrollY > 32);
});

