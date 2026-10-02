/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Canonical design tokens from the kickoff deck (Rashi Agarwal, App.css).
        // #f8fafc on #0f172a is ~17:1 contrast — past WCAG AAA (7:1).
        surface: '#0f172a',
        card: '#1e293b',
        text: '#f8fafc',
        accent: '#38bdf8',
        muted: '#94a3b8',
        line: '#334155',
        ok: '#4ade80',
        warn: '#fbbf24',
        bad: '#f87171',
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'ui-monospace', 'monospace'],
      },
      keyframes: {
        pulseRing: {
          '0%, 100%': { opacity: '0.45' },
          '50%': { opacity: '1' },
        },
        riseIn: {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        pulseRing: 'pulseRing 1.6s ease-in-out infinite',
        riseIn: 'riseIn 220ms ease-out',
      },
    },
  },
  plugins: [],
};
