import ProductCard from './ProductCard.jsx';

export default function ProductGrid({ products, onSelect, compact = false }) {
  return (
    <div className={`product-grid${compact ? ' product-grid-compact' : ''}`}>
      {products.map((product, index) => (
        <ProductCard
          key={product._id || `${product.name || 'product'}-${index}`}
          product={product}
          onSelect={onSelect}
          compact={compact}
        />
      ))}
    </div>
  );
}
