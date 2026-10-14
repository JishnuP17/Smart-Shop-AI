import { LoaderCircle, RotateCcw } from 'lucide-react';
import ProductGrid from './ProductGrid.jsx';

export default function DemoPersonalization({
  selectedUser,
  products,
  status,
  error,
  basedOn,
  onRetryRecommendations,
  onSelectProduct,
}) {
  return (
    <section className="explore-section demo-personalized" aria-labelledby="personalized-heading">
      <div className="demo-personalized-heading">
        <div className="explore-heading">
          <div>
            <span className="eyebrow">PERSONALIZED FOR {selectedUser?.name?.toUpperCase() || 'YOU'}</span>
            <h2 id="personalized-heading">Recommended for you</h2>
            <p>{basedOn === 'interests_and_interactions' ? 'Updated from this profile’s recent product activity.' : 'Selected from this profile’s interests.'}</p>
          </div>
        </div>
      </div>
      {status === 'loading' ? <div className="demo-inline-state" role="status"><LoaderCircle size={18} className="spin" /> Finding recommendations…</div> : null}
      {status === 'error' ? <div className="demo-inline-error" role="alert">{error}<button type="button" onClick={onRetryRecommendations}><RotateCcw size={14} /> Try again</button></div> : null}
      {status === 'ready' && products.length === 0 ? <p className="demo-inline-state">No recommendations are available for this profile yet.</p> : null}
      {status === 'ready' && products.length > 0 ? <ProductGrid products={products} onSelect={onSelectProduct} /> : null}
    </section>
  );
}
