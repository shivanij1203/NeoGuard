/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'neoguard-bg': '#0f172a',
        'neoguard-card': '#1e293b',
        'neoguard-border': '#334155',
      },
    },
  },
  plugins: [],
}
