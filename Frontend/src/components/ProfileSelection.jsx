import { ArrowRight, LoaderCircle, RotateCcw, ShoppingBag, Sparkles } from 'lucide-react';

export default function ProfileSelection({
  profiles = [],
  selectedId = '',
  onSelect = () => {},
  onContinue,
  loading = false,
  error = '',
  onRetry,
}) {
  return (
    <main className="profile-entry" aria-labelledby="profile-entry-title">
      <div className="profile-entry-glow" aria-hidden="true" />
      <div className="profile-entry-shell">
        <a className="profile-entry-brand" href="/" aria-label="Smart Shop home" onClick={(event) => event.preventDefault()}>
          <span><ShoppingBag size={19} /></span> SMART <b>SHOP</b>
        </a>
        <div className="profile-entry-copy">
          <span className="profile-entry-kicker"><Sparkles size={14} /> YOUR DEMO SHOPPING SESSION</span>
          <h1 id="profile-entry-title">Who’s shopping today?</h1>
          <p>Choose a demo profile to see product picks shaped around its interests. This is a shared-device demo profile, not a secure account.</p>
        </div>

        {loading ? (
          <div className="profile-entry-state" role="status"><LoaderCircle size={22} className="spin" /> Loading demo profiles…</div>
        ) : null}
        {error ? (
          <div className="profile-entry-error" role="alert">
            <p>{error}</p>
            {onRetry ? <button className="entry-retry" type="button" onClick={onRetry}><RotateCcw size={15} /> Try again</button> : null}
          </div>
        ) : null}

        {!loading && !error ? (
          <>
            <div className="profile-cards" role="group" aria-label="Choose a demo profile">
              {profiles.map((profile) => (
                <button
                  key={profile.id}
                  type="button"
                  className={`profile-card${selectedId === profile.id ? ' is-selected' : ''}`}
                  aria-pressed={selectedId === profile.id}
                  onClick={() => onSelect(profile.id)}
                >
                  <span className="profile-avatar" aria-hidden="true">{profile.name.slice(0, 1).toUpperCase()}</span>
                  <span className="profile-card-copy">
                    <strong>{profile.name}</strong>
                    <span>{profile.description}</span>
                    <span className="profile-interest-list">{profile.interests.join(' · ')}</span>
                  </span>
                  <span className="profile-choice-mark" aria-hidden="true">{selectedId === profile.id ? '✓' : ''}</span>
                </button>
              ))}
            </div>
            <button className="profile-continue" type="button" onClick={onContinue} disabled={!selectedId}>
              Enter Smart Shop <ArrowRight size={17} />
            </button>
          </>
        ) : null}
        <p className="profile-entry-footnote">Your selection is saved in this browser so it stays active after refresh.</p>
      </div>
    </main>
  );
}
