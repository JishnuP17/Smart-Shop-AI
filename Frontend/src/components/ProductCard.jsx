import { ArrowUpRight } from 'lucide-react';
import { useState } from 'react';

function formatPrice(price) {
  if (typeof price !== 'number' || !Number.isFinite(price)) return 'Price unavailable';
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' }).format(price);
}

export default function ProductCard({ product, onSelect, compact = false }) {
  const image = product.image_url || product.image;
  const [imageFailed, setImageFailed] = useState(false);
  return (
    <button
      type="button"
      className={`product-card${compact ? ' product-card-compact' : ''}`}
      onClick={() => onSelect(product)}
      aria-label={`View ${product.name || 'product'} details`}
    >
      <span className="product-image-wrap">
        {image && !imageFailed ? (
          <img src={image} alt={product.name || 'Product'} loading="lazy" onError={() => setImageFailed(true)} />
        ) : null}
        <span className="image-fallback" aria-hidden="true"><span>SS</span></span>
        <span className="card-open"><ArrowUpRight size={17} /></span>
      </span>
      <span className="product-card-info">
        <span className="product-card-topline">
          <span className="product-category">{product.category || 'Catalog find'}</span>
          {product.brand ? <span className="product-brand">{product.brand}</span> : null}
        </span>
        <span className="product-name">{product.name || 'Untitled product'}</span>
        <span className="product-card-bottom"><span className="product-price">{formatPrice(product.price)}</span><span className="card-arrow"><ArrowUpRight size={15} /></span></span>
      </span>
    </button>
  );
}
