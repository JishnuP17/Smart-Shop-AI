const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

async function request(path, { signal, method = 'GET', body } = {}) {
  const startedAt = performance.now();
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      signal,
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    if (signal?.aborted) throw new DOMException('Request was cancelled.', 'AbortError');
    throw new Error(`Could not reach the API at ${API_BASE_URL}. Check that the backend is running and the address is correct.`);
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      // Keep the useful status message when the server does not return JSON.
    }
    throw new Error(detail);
  }
  try {
    const body = await response.json();
    if (path.startsWith('/api/products/search')) {
      console.debug(`[Smart Shop] Search API request completed in ${(performance.now() - startedAt).toFixed(1)} ms (HTTP ${response.status}).`);
    }
    return body;
  } catch {
    throw new Error('Smart Shop returned an unreadable response. Please try again.');
  }
}

export function getProducts({ skip = 0, limit = 20, category, signal } = {}) {
  const params = new URLSearchParams({ skip: String(skip), limit: String(limit) });
  if (category) params.set('category', category);
  return request(`/api/products?${params}`, { signal });
}

export function getProductCategories(options = {}) {
  return request('/api/products/categories', options);
}

export function getProduct(productId, options = {}) {
  return request(`/api/products/${encodeURIComponent(productId)}`, options);
}

export function searchProducts(query, limit = 20, options = {}) {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return request(`/api/products/search?${params}`, options);
}

export function getRecommendations(productId, limit = 5) {
  const params = new URLSearchParams({ limit: String(limit) });
  return request(`/api/products/${encodeURIComponent(productId)}/recommendations?${params}`);
}

export function getDemoUsers(options = {}) {
  return request('/api/demo/users', options);
}

export function getPersonalizedRecommendations(userId, limit = 10, options = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  return request(`/api/demo/users/${encodeURIComponent(userId)}/recommendations?${params}`, options);
}

export function recordDemoInteraction(userId, interaction, options = {}) {
  return request(`/api/demo/users/${encodeURIComponent(userId)}/interactions`, {
    ...options,
    method: 'POST',
    body: interaction,
  });
}
