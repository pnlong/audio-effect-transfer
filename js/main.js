// Pause all other audio elements when one starts playing
document.addEventListener('DOMContentLoaded', () => {
  const allAudio = () => document.querySelectorAll('audio');

  document.body.addEventListener('play', (e) => {
    if (e.target.tagName !== 'AUDIO') return;
    allAudio().forEach((a) => {
      if (a !== e.target) a.pause();
    });
  }, true);
});
