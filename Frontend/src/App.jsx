import { ArrowRight, LoaderCircle, RotateCcw } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import ProfileSelection from './components/ProfileSelection.jsx';
import Header from './components/Header.jsx';
import DemoPersonalization from './components/DemoPersonalization.jsx';
import ProductDetails from './components/ProductDetails.jsx';
import ProductGrid from './components/ProductGrid.jsx';
import SearchBar from './components/SearchBar.jsx';
import {
  getDemoUsers,
  getPersonalizedRecommendations,
  getProduct,
  getProducts,
  recordDemoInteraction,
  searchProducts,
} from './services/api.js';

const PAGE_SIZE = 24;

function createEventId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function readSavedDemoUser() {
  try { return window.localStorage.getItem('smartshop-demo-session') || ''; } catch { return ''; }
}

function saveDemoUser(userId) {
  try {
    if (userId) window.localStorage.setItem('smartshop-demo-session', userId);
    else window.localStorage.removeItem('smartshop-demo-session');
  } catch { /* storage may be disabled */ }
}

function locationRoute() {
  const params = new URLSearchParams(window.location.search);
  const pathname = window.location.pathname;
  if (pathname === '/shop') return { page: 'shop', query: '' };
  if (pathname === '/search') return { page: 'search', query: params.get('q') || '' };
  const productMatch = pathname.match(/^\/product\/([^/]+)\/?$/);
  if (productMatch) return { page: 'product', productId: decodeURIComponent(productMatch[1]) };
  return { page: 'home', query: '' };
}

function StatePanel({ type, title, message, action, actionLabel }) {
  return (
    <div className={`state-panel ${type}-panel`} role={type === 'error' ? 'alert' : 'status'}>
      <span className="state-symbol" aria-hidden="true">{type === 'error' ? '!' : type === 'empty' ? '⌕' : null}</span>
      {type === 'loading' ? <LoaderCircle size={24} className="spin" /> : null}
      <h3>{title}</h3><p>{message}</p>
      {action ? <button type="button" className="primary-button" onClick={action}>{type === 'error' ? <RotateCcw size={15} /> : null}{actionLabel}</button> : null}
    </div>
  );
}

function Storefront({ demoUsers, selectedDemoUserId, onSwitchUser }) {
  const [route, setRoute] = useState(locationRoute);
  const [draft, setDraft] = useState(() => locationRoute().query);
  const [catalog, setCatalog] = useState([]);
  const [personalized, setPersonalized] = useState({ userId: '', products: [], status: 'loading', error: '', basedOn: '' });
  const [personalizationRetry, setPersonalizationRetry] = useState(0);
  const [personalizationRefresh, setPersonalizationRefresh] = useState(0);
  const [productVisitId, setProductVisitId] = useState(createEventId);
  const [searchResults, setSearchResults] = useState([]);
  const [resultCount, setResultCount] = useState(null);
  const [skip, setSkip] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState('');
  const [routedProduct, setRoutedProduct] = useState(null);
  const [productLoading, setProductLoading] = useState(false);
  const [productError, setProductError] = useState('');
  const activeRequest = useRef(null);
  const requestVersion = useRef(0);
  const recordedViewRef = useRef('');
  const searchInput = useRef(null);
  const routeRef = useRef(route);
  routeRef.current = route;

  const syncRoute = useCallback(() => {
    const next = locationRoute();
    setRoute(next);
    setDraft(next.query);
    setProductVisitId(createEventId());
  }, []);

  useEffect(() => {
    window.addEventListener('popstate', syncRoute);
    return () => window.removeEventListener('popstate', syncRoute);
  }, [syncRoute]);

  const navigate = useCallback((path) => {
    window.history.pushState(null, '', path);
    setProductVisitId(createEventId());
    syncRoute();
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, [syncRoute]);

  const sendInteraction = useCallback((event) => {
    if (!selectedDemoUserId) return;
    recordDemoInteraction(selectedDemoUserId, event)
      .then(() => setPersonalizationRefresh((current) => current + 1))
      .catch(() => { /* Recommendation browsing should continue if activity logging is unavailable. */ });
  }, [selectedDemoUserId]);

  const submitSearch = useCallback((value) => {
    const text = value.trim();
    if (text) sendInteraction({ event_id: createEventId(), event_type: 'search', query: text });
    navigate(`/search?q=${encodeURIComponent(text)}`);
  }, [navigate, sendInteraction]);

  useEffect(() => {
    if (!selectedDemoUserId || !demoUsers.some((user) => user.id === selectedDemoUserId)) return undefined;
    const controller = new AbortController();
    setPersonalized({ userId: selectedDemoUserId, products: [], status: 'loading', error: '', basedOn: '' });
    const requestTimer = window.setTimeout(() => {
      getPersonalizedRecommendations(selectedDemoUserId, 8, { signal: controller.signal })
        .then((response) => {
          if (controller.signal.aborted) return;
          setPersonalized({
            userId: selectedDemoUserId,
            products: (response.recommendations || []).map((hit) => hit.product).filter(Boolean),
            status: 'ready',
            error: '',
            basedOn: response.based_on || 'initial_interests',
          });
        })
        .catch((reason) => {
          if (reason.name !== 'AbortError' && !controller.signal.aborted) {
            setPersonalized({ userId: selectedDemoUserId, products: [], status: 'error', error: reason.message || 'Recommendations could not be loaded.', basedOn: '' });
          }
        });
    }, 0);
    return () => { window.clearTimeout(requestTimer); controller.abort(); };
  }, [selectedDemoUserId, demoUsers, personalizationRetry, personalizationRefresh]);

  const loadCatalog = useCallback(async ({ from = 0, append = false, limit = PAGE_SIZE } = {}) => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    const requestId = ++requestVersion.current;
    if (append) setLoadingMore(true);
    else setLoading(true);
    setError('');
    try {
      const response = await getProducts({ skip: from, limit, signal: controller.signal });
      if (requestId !== requestVersion.current) return;
      const items = Array.isArray(response.items) ? response.items : [];
      setCatalog((current) => append ? [...current, ...items] : items);
      setSkip(from + items.length);
      setHasMore(items.length === limit);
    } catch (reason) {
      if (reason.name === 'AbortError' || requestId !== requestVersion.current) return;
      setError(reason.message || 'The product catalog could not be loaded.');
      if (!append) setCatalog([]);
    } finally {
      if (requestId === requestVersion.current) {
        setLoading(false);
        setLoadingMore(false);
      }
    }
  }, []);

  useEffect(() => {
    if (route.page !== 'shop') return undefined;
    setSearchResults([]);
    setResultCount(null);
    loadCatalog();
    return () => activeRequest.current?.abort();
  }, [route.page, loadCatalog]);

  useEffect(() => {
    if (route.page !== 'product') return undefined;
    const controller = new AbortController();
    setRoutedProduct(null);
    setProductError('');
    setProductLoading(true);
    getProduct(route.productId, { signal: controller.signal })
      .then((product) => setRoutedProduct(product))
      .catch((reason) => {
        if (reason.name !== 'AbortError') setProductError(reason.message || 'This product could not be loaded.');
      })
      .finally(() => {
        if (!controller.signal.aborted) setProductLoading(false);
      });
    return () => controller.abort();
  }, [route.page, route.productId]);

  useEffect(() => {
    if (route.page !== 'product' || !routedProduct?._id || !selectedDemoUserId) return;
    const key = `${selectedDemoUserId}:${routedProduct._id}:${productVisitId}`;
    if (recordedViewRef.current === key) return;
    recordedViewRef.current = key;
    sendInteraction({ event_id: createEventId(), event_type: 'view', product_id: String(routedProduct._id) });
  }, [route.page, routedProduct, selectedDemoUserId, productVisitId, sendInteraction]);

  useEffect(() => {
    if (route.page !== 'search') return undefined;
    activeRequest.current?.abort();
    const requestId = ++requestVersion.current;
    setCatalog([]);
    setSearchResults([]);
    setResultCount(null);
    setError('');
    if (!route.query.trim()) {
      setLoading(false);
      return undefined;
    }
    const controller = new AbortController();
    activeRequest.current = controller;
    setLoading(true);
    // Deferring by one event-loop turn lets React StrictMode's development-only
    // setup/cleanup check cancel the first effect before it sends a duplicate GET.
    const requestTimer = window.setTimeout(() => {
      searchProducts(route.query, PAGE_SIZE, { signal: controller.signal })
        .then((response) => {
          if (requestId !== requestVersion.current) return;
          const hits = Array.isArray(response.results) ? response.results : [];
          setSearchResults(hits.map((hit) => hit.product).filter(Boolean));
          setResultCount(hits.length);
        })
        .catch((reason) => {
          if (reason.name === 'AbortError' || requestId !== requestVersion.current) return;
          setError(reason.message || 'Search is unavailable right now.');
        })
        .finally(() => {
          if (requestId === requestVersion.current) setLoading(false);
        });
    }, 0);
    return () => {
      window.clearTimeout(requestTimer);
      controller.abort();
    };
  }, [route.page, route.query]);

  function goHome() {
    navigate('/');
  }

  function goShop() {
    if (routeRef.current.page === 'shop') document.getElementById('products')?.scrollIntoView({ behavior: 'smooth' });
    else navigate('/shop');
  }

  function openProduct(product) {
    if (!product?._id) return;
    sendInteraction({ event_id: createEventId(), event_type: 'click', product_id: String(product._id) });
    navigate(`/product/${encodeURIComponent(product._id)}`);
  }

  function closeDetails() {
    navigate('/shop');
  }

  const isSearch = route.page === 'search';
  return (
    <div className="storefront storefront-dark" id="home">
      <Header onHome={goHome} onExplore={goShop} activePage={route.page} activeProfile={demoUsers.find((user) => user.id === selectedDemoUserId)} onSwitchUser={onSwitchUser} />
      <main>
        {route.page === 'home' ? (
          <>
            <section className="hero-section compact-hero" aria-labelledby="hero-title">
              <div className="hero-inner home-hero-inner">
                <div className="hero-copy">
                  <span className="hero-overline">SMART SHOP · PRODUCT DISCOVERY</span>
                  <h1 id="hero-title">Find what<br />fits <em>your life.</em></h1>
                  <p>Describe what you need in your own words and explore the catalog.</p>
                  <SearchBar query={draft} setQuery={setDraft} onSearch={submitSearch} onClear={() => setDraft('')} hasSearch={Boolean(draft)} inputRef={searchInput} busy={route.page === 'search' && loading} />
                  <button className="hero-link" type="button" onClick={goShop}>Browse products <ArrowRight size={15} /></button>
                </div>
              </div>
            </section>
            <DemoPersonalization
              selectedUser={demoUsers.find((user) => user.id === selectedDemoUserId)}
              products={personalized.userId === selectedDemoUserId ? personalized.products : []}
              status={personalized.userId === selectedDemoUserId ? personalized.status : 'loading'}
              error={personalized.userId === selectedDemoUserId ? personalized.error : ''}
              basedOn={personalized.userId === selectedDemoUserId ? personalized.basedOn : ''}
              onRetryRecommendations={() => setPersonalizationRetry((count) => count + 1)}
              onSelectProduct={openProduct}
            />
          </>
        ) : route.page === 'shop' ? (
          <section className="explore-section search-page shop-page" id="products" aria-labelledby="shop-heading">
            <div className="explore-heading"><div><span className="eyebrow">SMART SHOP CATALOG</span><h1 id="shop-heading">Shop all products</h1><p>Browse the live catalog, one page at a time.</p></div></div>
            {error ? <StatePanel type="error" title="We couldn’t load the products" message={error} action={() => loadCatalog()} actionLabel="Try again" /> : null}
            {!error && loading ? <StatePanel type="loading" title="Loading products" message="Fetching the latest catalog items." /> : null}
            {!error && !loading && catalog.length ? <ProductGrid products={catalog} onSelect={openProduct} /> : null}
            {!error && !loading && !catalog.length ? <StatePanel type="empty" title="No products are available" message="The catalog returned no products. Try again in a moment." action={() => loadCatalog()} actionLabel="Refresh catalog" /> : null}
            {!error && hasMore ? <div className="load-more-wrap"><button className="load-more" type="button" disabled={loadingMore} onClick={() => loadCatalog({ from: skip, append: true })}>{loadingMore ? <LoaderCircle size={16} className="spin" /> : null}{loadingMore ? 'Loading' : 'Load more products'}</button></div> : null}
          </section>
        ) : isSearch ? (
          <section className="explore-section search-page" id="products" aria-labelledby="results-heading">
            <div className="search-page-top">
              <div className="explore-heading"><div><span className="eyebrow">SMART SHOP SEARCH</span><h1 id="results-heading">Search results</h1><p>{route.query ? resultCount === null ? `Searching the catalog for “${route.query}”` : `${resultCount} products returned for “${route.query}”` : 'Enter a product or describe what you are looking for.'}</p></div></div>
              <SearchBar query={draft} setQuery={setDraft} onSearch={submitSearch} onClear={() => { setDraft(''); navigate('/search?q='); }} hasSearch={Boolean(draft)} inputRef={searchInput} compact busy={loading} />
            </div>
            {error ? <StatePanel type="error" title="Search could not be completed" message={error} action={() => submitSearch(route.query)} actionLabel="Try again" /> : null}
            {!error && route.query && loading ? <StatePanel type="loading" title="Finding relevant products" message="Searching the live product catalog." /> : null}
            {!error && !loading && !route.query ? <StatePanel type="empty" title="Start with a search" message="Enter a few words above, or return to the catalog to browse products." action={goHome} actionLabel="Browse the catalog" /> : null}
            {!error && !loading && route.query && searchResults.length ? <ProductGrid products={searchResults} onSelect={openProduct} /> : null}
            {!error && !loading && route.query && !searchResults.length ? <StatePanel type="empty" title="No products returned" message="The search API returned no matches. Try a different description or edit your query above." /> : null}
            {!error && !loading && searchResults.length ? <p className="ranking-note">Results are shown in the backend’s semantic relevance order.</p> : null}
          </section>
        ) : (
          <section className="explore-section search-page product-route" aria-label="Product details">
            {productLoading ? <StatePanel type="loading" title="Loading product" message="Fetching this product from the live catalog." /> : null}
            {!productLoading && productError ? <StatePanel type="error" title="Product unavailable" message={productError} action={goShop} actionLabel="Browse the catalog" /> : null}
            {!productLoading && !productError && routedProduct ? <ProductDetails key={routedProduct._id} product={routedProduct} onClose={closeDetails} onSelect={openProduct} /> : null}
          </section>
        )}
      </main>
      <footer className="store-footer"><button className="footer-brand" type="button" onClick={goHome}>Smart Shop</button><span>Thoughtful finds from the product catalog.</span><button className="footer-top" type="button" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>Back to top ↑</button></footer>
    </div>
  );
}

export default function App() {
  const [demoUsers, setDemoUsers] = useState([]);
  const [profilesLoading, setProfilesLoading] = useState(true);
  const [profilesError, setProfilesError] = useState('');
  const [profilesRetry, setProfilesRetry] = useState(0);
  const [activeUserId, setActiveUserId] = useState(() => readSavedDemoUser());
  const [pendingUserId, setPendingUserId] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    setProfilesLoading(true);
    setProfilesError('');
    const requestTimer = window.setTimeout(() => {
      getDemoUsers({ signal: controller.signal })
        .then((response) => {
          if (controller.signal.aborted) return;
          const profiles = Array.isArray(response.items) ? response.items : [];
          setDemoUsers(profiles);
          setActiveUserId((current) => {
            if (current && profiles.some((profile) => profile.id === current)) return current;
            saveDemoUser('');
            return '';
          });
        })
        .catch((reason) => {
          if (reason.name !== 'AbortError' && !controller.signal.aborted) setProfilesError(reason.message || 'Demo profiles could not be loaded.');
        })
        .finally(() => { if (!controller.signal.aborted) setProfilesLoading(false); });
    }, 0);
    return () => { window.clearTimeout(requestTimer); controller.abort(); };
  }, [profilesRetry]);

  function continueAsSelectedUser() {
    if (!pendingUserId || !demoUsers.some((profile) => profile.id === pendingUserId)) return;
    saveDemoUser(pendingUserId);
    setActiveUserId(pendingUserId);
  }

  function switchUser() {
    saveDemoUser('');
    setActiveUserId('');
    setPendingUserId('');
  }

  if (profilesLoading) {
    return <ProfileSelection loading profiles={demoUsers} selectedId={pendingUserId} onSelect={setPendingUserId} />;
  }
  if (profilesError || !demoUsers.length) {
    return <ProfileSelection error={profilesError || 'No demo profiles are available.'} profiles={demoUsers} selectedId={pendingUserId} onSelect={setPendingUserId} onRetry={() => setProfilesRetry((value) => value + 1)} />;
  }
  if (!activeUserId) {
    return <ProfileSelection profiles={demoUsers} selectedId={pendingUserId} onSelect={setPendingUserId} onContinue={continueAsSelectedUser} />;
  }
  return <Storefront key={activeUserId} demoUsers={demoUsers} selectedDemoUserId={activeUserId} onSwitchUser={switchUser} />;
}
