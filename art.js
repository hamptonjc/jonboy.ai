(() => {
  'use strict';
  const canvas = document.getElementById('field');
  const ctx = canvas.getContext('2d');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const button = document.getElementById('motion');
  let paused = reduced.matches, visible = true, time = 0, last = 0, frame = 0;
  let width = 1, height = 1, pointer = 0, target = 0;
  const resize = () => {
    const box = canvas.getBoundingClientRect();
    width = box.width; height = box.height;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
  };
  function draw() {
    ctx.clearRect(0, 0, width, height);
    const mobile = window.matchMedia('(max-width: 760px)').matches;
    const scale = Math.min(width / (mobile ? 600 : 1120), height / 750);
    ctx.save();ctx.translate(width * (mobile ? .5 : .58) + pointer * 10, height * .44);ctx.scale(scale, scale);
    // Nested, folded curves: an original generative study of emergence.
    for (let i = 0; i < 112; i++) {
      const layer = i / 111, radius = 48 + layer * 355;
      ctx.beginPath();
      for (let j = 0; j <= 420; j++) {
        const a = j / 420 * Math.PI * 2;
        const fold = 1 + .22 * Math.cos(a * 3 + layer * 5 + time * .12) + .10 * Math.sin(a * 5 - layer * 3 - time * .08);
        const x = Math.cos(a) * radius * fold;
        const y = Math.sin(a) * radius * fold * .75;
        const twist = -.45 + layer * .55 + Math.sin(time * .07) * .08;
        const px = x * Math.cos(twist) - y * Math.sin(twist);
        const py = x * Math.sin(twist) + y * Math.cos(twist);
        if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();ctx.strokeStyle = `rgba(185,207,158,${.10 + .36 * Math.sin(layer * Math.PI)})`;ctx.lineWidth = .65;ctx.stroke();
    }
    ctx.restore();
  }
  function animate(timestamp) {
    frame = 0;
    if (paused || !visible || document.hidden) return;
    if (timestamp - last > 42) { time += .035; pointer += (target - pointer) * .025; draw(); last = timestamp; }
    frame = requestAnimationFrame(animate);
  }
  function sync() {
    button.setAttribute('aria-pressed', String(paused));
    button.innerHTML = paused ? '<span class="motion-icon">▷</span> Play motion' : '<span class="motion-icon">Ⅱ</span> Pause motion';
    if (frame) cancelAnimationFrame(frame);frame = 0;
    if (!paused && visible && !document.hidden) frame = requestAnimationFrame(animate);
  }
  button.addEventListener('click', () => {paused = !paused;sync();});
  reduced.addEventListener('change', e => {paused = e.matches;sync();});
  document.addEventListener('visibilitychange', sync);
  document.querySelector('.hero').addEventListener('pointermove', e => {target = e.clientX / window.innerWidth - .5;});
  document.querySelector('.hero').addEventListener('pointerleave', () => {target = 0;});
  new IntersectionObserver(entries => {visible = entries[0].isIntersecting;sync();}).observe(canvas);
  new ResizeObserver(resize).observe(canvas);
  const resumeDisclosure = document.querySelector('#resume > details');
  const openSection = hash => {
    if (hash !== '#resume' && hash !== '#work') return;
    document.querySelector(`${hash} > details`).open = true;
  };
  document.querySelectorAll('a[href="#resume"], a[href="#work"]').forEach(link => {
    link.addEventListener('click', () => openSection(link.getAttribute('href')));
  });
  window.addEventListener('hashchange', () => openSection(window.location.hash));
  openSection(window.location.hash);
  let resumeOpenBeforePrint = false;
  window.addEventListener('beforeprint', () => {
    resumeOpenBeforePrint = resumeDisclosure.open;
    resumeDisclosure.open = true;
  });
  window.addEventListener('afterprint', () => { resumeDisclosure.open = resumeOpenBeforePrint; });
  document.getElementById('year').textContent = new Date().getFullYear();
  resize();sync();
})();
