import { ArrowLeft, ArrowUpRight, LoaderCircle, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { getRecommendations } from '../services/api.js';
import ProductGrid from './ProductGrid.jsx';

function formatPrice(price) {
  if (typeof price !== 'number' || !Number.isFinite(price)) return 'Price unavailable';
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' }).format(price);
}

export default function ProductDetails({ product, onClose, onSelect }) {
  const [recommendations, setRecommendations] = useState([]);
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState('');
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    let active = true;
    setRecommendations([]);
    setError('');
    if (!product?._id) {
      setStatus('error');
      setError('This product does not have an ID for recommendations.');
      return () => { active = false; };
    }
    setStatus('loading');
    getRecommendations(product._id, 5)
      .then((result) => {
        if (!active) return;
        setRecommendations(Array.isArray(result.recommendations) ? result.recommendations.map((hit) => hit.product) : []);
        setStatus('ready');
      })
      .catch((reason) => {
        if (!active) return;
        setError(reason.message || 'Recommendations are unavailable right now.');
        setStatus('error');
      });
    return () => { active = false; };
  }, [product?._id]);

  useEffect(() => { setImageFailed(false); }, [product?._id]);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const image = product.image_url || product.image;
  return (
    <div className="detail-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="detail-panel" role="dialog" aria-modal="true" aria-labelledby="detail-name">
        <div className="detail-topbar">
          <button className="back-button" type="button" onClick={onClose}><ArrowLeft size={17} /> Back to browsing</button>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close product details"><X size={20} /></button>
        </div>
        <div className="detail-main">
          <div className="detail-image">
            {image && !imageFailed ? <img src={image} alt={product.name || 'Product'} onError={() => setImageFailed(true)} /> : null}
            <span className="image-fallback" aria-hidden="true"><span>SS</span></span>
          </div>
          <div className="detail-copy">
            <span className="eyebrow">{product.category || 'A considered find'}</span>
            <h2 id="detail-name">{product.name || 'Untitled product'}</h2>
            <p className="detail-price">{formatPrice(product.price)}</p>
            {(product.brand || product.category) && (
              <dl className="detail-meta">
                {product.brand ? <div><dt>Brand</dt><dd>{product.brand}</dd></div> : null}
                {product.category ? <div><dt>Category</dt><dd>{product.category}</dd></div> : null}
              </dl>
            )}
            {product.description ? <p className="detail-description">{product.description}</p> : <p className="detail-description muted">No description is available for this product.</p>}
            {image ? <a className="source-link" href={image} target="_blank" rel="noreferrer">View product image <ArrowUpRight size={15} /></a> : null}
          </div>
        </div>
        <section className="recommendations-section" aria-labelledby="similar-heading">
          <div className="section-heading compact-heading">
            <div><span className="eyebrow">Picked from the catalog</span><h3 id="similar-heading">Similar products</h3></div>
          </div>
          {status === 'loading' ? <div className="inline-state"><LoaderCircle className="spin" size={19} /> Finding similar products…</div> : null}
          {status === 'error' ? <div className="inline-error">{error}</div> : null}
          {status === 'ready' && recommendations.length === 0 ? <p className="empty-note">No similar products were found.</p> : null}
          {recommendations.length > 0 ? <ProductGrid products={recommendations} onSelect={onSelect} compact /> : null}
        </section>
      </section>
    </div>
  );
}
