(function () {
  const toggle = document.getElementById('chat-toggle');
  const panel = document.getElementById('chat-panel');
  const closeBtn = document.getElementById('chat-close');
  const body = document.getElementById('chat-body');
  const form = document.getElementById('chat-form');
  const text = document.getElementById('chat-text');
  const suggest = document.getElementById('chat-suggest');
  if (!toggle || !panel) return;

  let greeted = false;

  function add(role, msg) {
    const el = document.createElement('div');
    el.className = 'chat-msg ' + role;
    el.textContent = msg;
    body.appendChild(el);
    body.scrollTop = body.scrollHeight;
    return el;
  }

  async function send(message, showUser) {
    if (showUser) add('user', message);
    const typing = add('bot typing', '·  ·  ·');
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      const data = await res.json();
      typing.remove();
      add('bot', data.reply || 'Sorry, something went wrong.');
    } catch (e) {
      typing.remove();
      add('bot', 'Network error — please try again.');
    }
  }

  function open() {
    panel.hidden = false;
    toggle.classList.add('open');
    if (!greeted) { greeted = true; send('hello', false); }
    text.focus();
  }
  function close() { panel.hidden = true; toggle.classList.remove('open'); }

  toggle.addEventListener('click', () => (panel.hidden ? open() : close()));
  closeBtn.addEventListener('click', close);
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const m = text.value.trim();
    if (!m) return;
    text.value = '';
    send(m, true);
  });
  suggest.addEventListener('click', (e) => {
    const q = e.target.dataset.q;
    if (!q) return;
    if (panel.hidden) open();
    send(q, true);
  });
})();
