import { Search, Sparkles, X } from 'lucide-react';

const examples = ['quiet headphones for commuting', 'a thoughtful gift under $50'];

export default function SearchBar({ query, setQuery, onSearch, onClear, hasSearch, busy = false, inputRef, compact = false }) {
  function submit(event) {
    event.preventDefault();
    if (busy) return;
    onSearch(query);
  }

  return (
    <div className={`search-wrap${compact ? ' search-wrap-compact' : ''}`} id="search">
      <form className="search-form" onSubmit={submit}>
        <Search className="search-icon" size={20} aria-hidden="true" />
        <label className="sr-only" htmlFor="product-search">Describe the product you are looking for</label>
        <input
          id="product-search"
          ref={inputRef}
          autoComplete="off"
          aria-busy={busy}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Describe the product you're looking for..."
          maxLength={500}
        />
        {hasSearch && !busy ? <button className="clear-search" type="button" onClick={onClear} aria-label="Clear search"><X size={17} /></button> : null}
        <button type="submit" disabled={busy || !query.trim()}>
          {busy ? <span className="button-spinner" /> : <Sparkles size={16} />}
          <span>{busy ? 'Searching' : 'Search'}</span>
        </button>
      </form>
      <div className="search-examples">
        <span>Try</span>
        {examples.map((example) => (
          <button key={example} type="button" disabled={busy} onClick={() => { setQuery(example); onSearch(example); }}>
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
