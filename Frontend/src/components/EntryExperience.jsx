import { ArrowUpRight, ShoppingBag } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

const particles = Array.from({ length: 18 }, (_, index) => index);

export default function EntryExperience({ onEnter }) {
  const [entering, setEntering] = useState(false);
  const timeoutRef = useRef(null);

  useEffect(() => () => window.clearTimeout(timeoutRef.current), []);

  function enterWithTransition() {
    if (entering) return;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reducedMotion) {
      onEnter();
      return;
    }
    setEntering(true);
    timeoutRef.current = window.setTimeout(onEnter, 520);
  }

  return (
    <main className={`entry-experience${entering ? ' entry-experience-leaving' : ''}`}>
      <div className="entry-grid" aria-hidden="true" />
      <div className="entry-orbit entry-orbit-one" aria-hidden="true" />
      <div className="entry-orbit entry-orbit-two" aria-hidden="true" />
      <div className="entry-light-sweep" aria-hidden="true" />
      <div className="entry-particles" aria-hidden="true">
        {particles.map((particle) => (
          <span
            key={particle}
            style={{
              '--particle-x': `${(particle * 61 + 7) % 100}%`,
              '--particle-y': `${(particle * 37 + 11) % 100}%`,
              '--particle-duration': `${8 + particle * 0.37}s`,
              '--particle-delay': `${particle * -0.6}s`,
            }}
          />
        ))}
      </div>

      <section className="entry-content" aria-labelledby="entry-title">
        <div className="entry-wordmark" aria-label="Smart Shop">
          <span className="entry-mark"><ShoppingBag size={21} strokeWidth={1.7} /></span>
          <span>SMART <b>SHOP</b></span>
        </div>
        <div className="entry-kicker"><span /> A NEW WAY TO DISCOVER</div>
        <h1 id="entry-title">SHOP BEYOND<br /><em>THE ORDINARY.</em></h1>
        <p className="entry-tagline">Thoughtful finds. Unexpected favorites.<br />Your next discovery starts here.</p>
        <button className="entry-enter" type="button" onClick={enterWithTransition} disabled={entering}>
          <span>{entering ? 'ENTERING SMART SHOP' : 'ENTER SMART SHOP'}</span>
          <ArrowUpRight size={19} strokeWidth={1.8} />
        </button>
      </section>

      <div className="entry-footer" aria-hidden="true"><span>SMART SHOP</span><span>DISCOVER WITH INTENTION</span><span>01 — 01</span></div>
    </main>
  );
}
