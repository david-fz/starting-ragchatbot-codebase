# Frontend Changes: Dark/Light Theme Toggle

## Summary
Added a toggle button that switches the UI between the existing dark theme and a new light theme, with the choice persisted across page reloads.

## Files Changed

### `frontend/index.html`
- Added a fixed-position `#themeToggle` button (top-right of the viewport) containing two inline SVG icons (sun and moon). It's a native `<button>` so it's keyboard-focusable and activates on Enter/Space out of the box.
- Includes `aria-label` and `aria-pressed` attributes, both kept in sync with the current theme via JS.
- Bumped the `style.css`/`script.js` cache-busting query params from `v=10` to `v=11`.

### `frontend/style.css`
- Added a `:root[data-theme="light"]` block redefining all existing CSS custom properties (`--background`, `--surface`, `--surface-hover`, `--text-primary`, `--text-secondary`, `--border-color`, `--assistant-message`, `--shadow`, `--focus-ring`, `--welcome-bg`) with light, accessible values (dark text on light backgrounds, same primary blue accent for brand consistency).
- Added a `transition` rule across the key themed elements (body, sidebar, chat panes, input, buttons, message bubbles, source chips) so switching themes animates smoothly instead of snapping.
- Added `.theme-toggle` styles: a circular button fixed to the top-right, using existing surface/border/shadow tokens so it matches the app's aesthetic in both themes; hover/focus-visible/active states follow the same patterns as other interactive elements (`#sendButton`, `.suggested-item`).
- Added crossfade + rotate transition between the sun and moon icons based on `[data-theme="light"]`.
- Added a small media-query tweak to shrink the button on narrow viewports.

### `frontend/script.js`
- Added `themeToggle` to the cached DOM elements and wired a `click` listener to `toggleTheme`.
- Added `initTheme()` (called on load): reads a saved theme from `localStorage`, falling back to the OS-level `prefers-color-scheme` if nothing is saved, then applies it.
- Added `toggleTheme()` / `applyTheme(theme)`: sets or removes `data-theme="light"` on `<html>`, persists the choice to `localStorage`, and updates the button's `aria-label`/`aria-pressed` for screen readers.

## Behavior
- Default/no preference: dark theme (unchanged existing look).
- Click (or keyboard-activate) the sun/moon button in the top-right to switch themes; the choice is remembered via `localStorage` and re-applied on the next visit.
- All existing components (messages, sidebar, inputs, source chips, suggested questions) inherit the new theme automatically since they were already built on CSS variables.
