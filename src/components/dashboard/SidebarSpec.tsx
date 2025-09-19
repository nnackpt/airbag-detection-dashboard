// File: src/components/dashboard/SidebarSpec.tsx
export default function SidebarSpec() {
  return (
    <div>
      {/* Wordmark */}
      <div className="side-wordmark">Autoliv</div>

      {/* SPEC title */}
      <div className="side-title">SPEC</div>
      <div className="divider" />

      {/* AMBIENT */}
      <div className="spec-block">
        <div className="spec-head">AMBIENT</div>
        <div className="spec-row ambient">
          <div className="pill">
            Target<br/>FRONT#1<br/><small>Spec: ≤ 17 ms</small>
          </div>
          <div className="pill">
            Target<br/>FRONT#2<br/><small>Spec: ≤ 20 ms</small>
          </div>
          <div className="pill">
            Target<br/>REAR#3<br/><small>Spec: ≤ 19 ms</small>
          </div>
        </div>
      </div>

      <div className="divider" />

      {/* HOT */}
      <div className="spec-block">
        <div className="spec-head">HOT</div>
        <div className="spec-row hot">
          <div className="pill">
            Target<br/>FRONT#1<br/><small>Spec: ≤ 17 ms</small>
          </div>
          <div className="pill">
            Target<br/>FRONT#2<br/><small>Spec: ≤ 20 ms</small>
          </div>
          <div className="pill">
            Target<br/>REAR#3<br/><small>Spec: ≤ 19 ms</small>
          </div>
        </div>
      </div>

      <div className="divider" />

      {/* COLD */}
      <div className="spec-block">
        <div className="spec-head">COLD</div>
        <div className="spec-row cold">
          <div className="pill">
            Target<br/>FRONT#1<br/><small>Spec: ≤ 21 ms</small>
          </div>
          <div className="pill">
            Target<br/>FRONT#2<br/><small>Spec: ≤ 22 ms</small>
          </div>
          <div className="pill">
            Target<br/>REAR#3<br/><small>Spec: ≤ 20 ms</small>
          </div>
        </div>
      </div>
    </div>
  );
}
