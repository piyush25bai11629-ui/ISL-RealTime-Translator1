import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Text-to-speech via the Web Speech API.
 *
 * The research deck ends the pipeline at "Text + TTS". Coqui TTS is listed as a
 * server-side option; speechSynthesis is used here because it costs nothing,
 * adds no latency to the inference loop and works offline in every target
 * browser. If the backend later returns audio, swap the body of speak().
 */
export function useTTS() {
  const [supported, setSupported] = useState(false);
  const [enabled, setEnabled] = useState(true);
  const [speaking, setSpeaking] = useState(false);
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    setSupported(true);

    const pick = () => {
      const voices = window.speechSynthesis.getVoices();
      if (!voices.length) return;
      // Prefer an Indian English voice, then any English voice.
      voiceRef.current =
        voices.find((v) => v.lang === 'en-IN') ??
        voices.find((v) => v.lang.startsWith('en')) ??
        voices[0];
    };

    pick();
    window.speechSynthesis.addEventListener('voiceschanged', pick);
    return () => {
      window.speechSynthesis.removeEventListener('voiceschanged', pick);
      window.speechSynthesis.cancel();
    };
  }, []);

  const speak = useCallback(
    // `force` is used by the explicit "Speak" button, which should work even
    // when automatic announcements are switched off.
    (text: string, force = false) => {
      if ((!enabled && !force) || !text.trim()) return;
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;

      const utter = new SpeechSynthesisUtterance(text);
      if (voiceRef.current) utter.voice = voiceRef.current;
      utter.rate = 0.95;
      utter.pitch = 1;
      utter.onstart = () => setSpeaking(true);
      utter.onend = () => setSpeaking(false);
      utter.onerror = () => setSpeaking(false);

      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
    },
    [enabled],
  );

  const stop = useCallback(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setSpeaking(false);
  }, []);

  return { supported, enabled, setEnabled, speaking, speak, stop };
}
