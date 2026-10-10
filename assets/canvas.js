export default function(component) {
  const { parentElement, data, setStateValue, setTriggerValue } = component;
  const canvas = parentElement.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const clearButton = parentElement.querySelector('.clear');
  const predictButton = parentElement.querySelector('.predict');
  const black = () => { ctx.save(); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 280, 280); ctx.restore(); };
  if (!canvas.dataset.initialized) { black(); canvas.dataset.initialized = '1'; }
  const incoming = data.image || '';
  if (incoming !== (canvas.dataset.png || '')) {
    canvas.dataset.png = incoming;
    if (incoming) {
      const image = new Image();
      image.onload = () => {
        if (canvas.dataset.png === incoming) { black(); ctx.drawImage(image, 0, 0, 280, 280); }
      };
      image.src = incoming;
    } else { black(); }
  }
  ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff';
  ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  let drawing = false;
  const point = event => {
    const r = canvas.getBoundingClientRect();
    return [(event.clientX-r.left)*280/r.width, (event.clientY-r.top)*280/r.height];
  };
  const commit = () => {
    const png = canvas.toDataURL('image/png');
    canvas.dataset.png = png;
    setStateValue('drawing', png);
  };
  canvas.onpointerdown = event => {
    if (event.button !== 0 && event.pointerType === 'mouse') return;
    event.preventDefault(); canvas.setPointerCapture(event.pointerId); drawing = true;
    const [x,y] = point(event); ctx.beginPath(); ctx.arc(x,y,9,0,2*Math.PI); ctx.fill();
    ctx.beginPath(); ctx.moveTo(x,y);
  };
  canvas.onpointermove = event => {
    if (!drawing) return;
    event.preventDefault(); const [x,y] = point(event); ctx.lineTo(x,y); ctx.stroke();
  };
  canvas.onpointerup = event => {
    if (!drawing) return;
    drawing = false; ctx.closePath();
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    commit();
  };
  canvas.onpointercancel = () => { if (drawing) { drawing = false; commit(); } };
  clearButton.onclick = () => {
    drawing = false; black(); canvas.dataset.png = '';
    setStateValue('drawing', null); setTriggerValue('clear', true);
  };
  predictButton.disabled = !data.can_predict;
  predictButton.onclick = () => {
    // Snapshot directly on click so Predict always uses the latest visible drawing.
    const png = canvas.toDataURL('image/png');
    canvas.dataset.png = png;
    setTriggerValue('predict', {png, nonce: Date.now()});
  };
  return () => {
    canvas.onpointerdown = null; canvas.onpointermove = null;
    canvas.onpointerup = null; canvas.onpointercancel = null;
    clearButton.onclick = null; predictButton.onclick = null;
  };
}
