import { ChevronDown, ShoppingBag, UserRound } from 'lucide-react';

export default function Header({ onHome, onExplore, activePage, activeProfile, onSwitchUser }) {
  return (
    <header className="topbar">
      <a className="wordmark" href="/" onClick={(event) => { event.preventDefault(); onHome(); }} aria-label="Smart Shop home">
        <span className="wordmark-icon"><ShoppingBag size={18} strokeWidth={1.8} /></span>
        <span>Smart <strong>Shop</strong></span>
      </a>
      <nav className="main-nav" aria-label="Main navigation">
        <a href="/" aria-current={activePage === 'home' ? 'page' : undefined} onClick={(event) => { event.preventDefault(); onHome(); }}>Home</a>
        <a href="/shop" aria-current={activePage === 'shop' ? 'page' : undefined} onClick={(event) => { event.preventDefault(); onExplore(); }}>Shop</a>
        <details className="profile-menu">
          <summary aria-label={`Demo profile: ${activeProfile?.name || 'active'}`}>
            <UserRound size={16} /> <span>{activeProfile?.name || 'Demo profile'}</span> <ChevronDown size={14} />
          </summary>
          <div className="profile-menu-panel">
            <span className="profile-menu-label">SHOPPING AS</span>
            <strong>{activeProfile?.name || 'Demo profile'}</strong>
            <p>{activeProfile?.description || 'A temporary demo profile for this browser.'}</p>
            <button type="button" onClick={(event) => {
              event.currentTarget.closest('details')?.removeAttribute('open');
              onSwitchUser();
            }}>Switch user</button>
          </div>
        </details>
      </nav>
    </header>
  );
}
