import Std

/-!
# CompactFlow: conditional safety of immutable field publication

This is an abstract transition-system proof, not a verification of Python.
External results are fixed by `Model.eval`; its `exact` hypothesis is the trusted
footprint/contract boundary. `Reference` specifies a complete-dependency result.
The proof does not establish annotation inference accuracy, eventual termination,
retry/compensation correctness, or the empirical violation rate.
-/
namespace CompactFlow

abbrev Call := Nat
abbrev Field := Nat
abbrev Resource := Nat

structure Model (Value : Type) where
  footprint : Call → List Field
  owner : Field → Call
  produces : Call → List Field
  eval : Call → (Field → Value) → Field → Value
  exact : ∀ c a b, (∀ f ∈ footprint c, a f = b f) → eval c a = eval c b
  earlySafe : Call → Prop
  effectBefore : Call → List Call
  demand : Call → Resource → Nat
  capacity : Resource → Nat

structure State (Value : Type) where
  store : Field → Option Value
  arguments : Call → Field → Value
  result : Call → Field → Value
  started : List Call
  active : List Call
  finished : List Call

def put {α : Type} (old : Nat → α) (k : Nat) (v : α) : Nat → α :=
  fun x => if x = k then v else old x

@[simp] theorem put_same {α : Type} (a : Nat → α) (k : Nat) (v : α) :
    put a k v k = v := by simp [put]

@[simp] theorem put_other {α : Type} (a : Nat → α) (k x : Nat) (v : α)
    (h : x ≠ k) : put a k v x = a x := by simp [put, h]

def initial {Value : Type} [Inhabited Value] : State Value :=
  ⟨fun _ => none, fun _ _ => default, fun _ _ => default, [], [], []⟩

def view {Value : Type} [Inhabited Value] (s : State Value) : Field → Value :=
  fun f => (s.store f).getD default

def load {Value : Type} (m : Model Value) : List Call → Resource → Nat
  | [], _ => 0
  | c :: cs, r => m.demand c r + load m cs r

def remove (c : Call) : List Call → List Call
  | [] => []
  | a :: xs => if a = c then remove c xs else a :: remove c xs

theorem load_remove_le {Value : Type} (m : Model Value) (c : Call)
    (xs : List Call) (r : Resource) : load m (remove c xs) r ≤ load m xs r := by
  induction xs with
  | nil => simp [remove, load]
  | cons a xs ih =>
    by_cases h : a = c
    · simp only [remove, h, if_true, load]
      omega
    · simp only [remove, h, if_false, load]
      omega

/-- A fixed complete-dependency result, including the same external responses. -/
def Reference {Value : Type} (m : Model Value) (ref : Field → Value) : Prop :=
  ∀ c f, m.owner f = c → m.eval c ref f = ref f

inductive Event where
  | dispatch (c : Call)
  | publish (f : Field)
  | complete (c : Call)
  deriving DecidableEq, Repr

/-- Every publication is a complete immutable field, never a raw token prefix.
    Completion abstracts the point after all declared final fields are committed.
    This fragment models successful single-attempt calls; failure is not success. -/
inductive Step {Value : Type} [Inhabited Value] (m : Model Value) :
    State Value → Event → State Value → Prop where
  | dispatch (s : State Value) (c : Call)
      (fresh : c ∉ s.started)
      (ready : ∀ f ∈ m.footprint c, ∃ v, s.store f = some v)
      (early : ∀ f ∈ m.footprint c, m.owner f ∈ s.finished ∨ m.earlySafe c)
      (effects : ∀ p ∈ m.effectBefore c, p ∈ s.finished)
      (fits : ∀ r, load m (c :: s.active) r ≤ m.capacity r) :
      Step m s (.dispatch c)
        { s with arguments := put s.arguments c (view s),
                 result := put s.result c (m.eval c (view s)),
                 started := c :: s.started, active := c :: s.active }
  | publish (s : State Value) (f : Field)
      (running : m.owner f ∈ s.active)
      (started : m.owner f ∈ s.started)
      (declared : f ∈ m.produces (m.owner f))
      (unpublished : s.store f = none) :
      Step m s (.publish f)
        { s with store := put s.store f (some (s.result (m.owner f) f)) }
  | complete (s : State Value) (c : Call)
      (running : c ∈ s.active)
      (fields : ∀ f ∈ m.produces c, ∃ v, s.store f = some v) :
      Step m s (.complete c)
        { s with active := remove c s.active, finished := c :: s.finished }

structure Invariant {Value : Type} (m : Model Value) (ref : Field → Value)
    (s : State Value) : Prop where
  values : ∀ f v, s.store f = some v → v = ref f
  results : ∀ c ∈ s.started, ∀ f, m.owner f = c → s.result c f = ref f
  arguments : ∀ c ∈ s.started, ∀ f ∈ m.footprint c, s.arguments c f = ref f
  unique : s.started.Nodup
  bounded : ∀ r, load m s.active r ≤ m.capacity r

theorem initial_invariant {Value : Type} [Inhabited Value]
    (m : Model Value) (ref : Field → Value) : Invariant m ref initial := by
  constructor
  · intro f v h; simp [initial] at h
  · intro c h; simp [initial] at h
  · intro c h; simp [initial] at h
  · simp [initial]
  · intro r; simp [initial, load]

theorem step_preserves {Value : Type} [Inhabited Value]
    {m : Model Value} {ref : Field → Value} (reference : Reference m ref)
    {s t : State Value} {event : Event}
    (inv : Invariant m ref s) (step : Step m s event t) : Invariant m ref t := by
  cases step with
  | dispatch c fresh ready early effects fits =>
    have input_eq : ∀ f ∈ m.footprint c, view s f = ref f := by
      intro f hf
      obtain ⟨v, hv⟩ := ready f hf
      simp only [view, hv, Option.getD_some]
      exact inv.values f v hv
    have evaluated : m.eval c (view s) = m.eval c ref := m.exact c _ _ input_eq
    constructor
    · exact inv.values
    · intro d hd f hf
      rcases List.mem_cons.mp hd with h | h
      · have owner : m.owner f = c := hf.trans h
        simpa only [h, put_same, evaluated] using reference c f owner
      · have ne : d ≠ c := by intro e; exact fresh (e ▸ h)
        simp only [put_other _ _ _ _ ne]
        exact inv.results d h f hf
    · intro d hd f hf
      rcases List.mem_cons.mp hd with h | h
      · subst d; simpa using input_eq f hf
      · have ne : d ≠ c := by intro e; exact fresh (e ▸ h)
        simp only [put_other _ _ _ _ ne]
        exact inv.arguments d h f hf
    · exact List.nodup_cons.mpr ⟨fresh, inv.unique⟩
    · exact fits
  | publish f running started declared unpublished =>
    constructor
    · intro g v hg
      by_cases h : g = f
      · subst g
        simp only [put_same, Option.some.injEq] at hg
        rw [← hg]
        exact inv.results (m.owner f) started f rfl
      · simp only [put_other _ _ _ _ h] at hg
        exact inv.values g v hg
    · exact inv.results
    · exact inv.arguments
    · exact inv.unique
    · exact inv.bounded
  | complete c running fields =>
    constructor
    · exact inv.values
    · exact inv.results
    · exact inv.arguments
    · exact inv.unique
    · intro r
      exact Nat.le_trans (load_remove_le m c s.active r) (inv.bounded r)

/-- Reachability is defined by transitions, not by assuming the desired result. -/
inductive Run {Value : Type} [Inhabited Value] (m : Model Value) :
    State Value → List Event → State Value → Prop where
  | nil (s) : Run m s [] s
  | cons {s t u e es} : Step m s e t → Run m t es u → Run m s (e :: es) u

theorem run_preserves {Value : Type} [Inhabited Value]
    {m : Model Value} {ref : Field → Value} (reference : Reference m ref)
    {s t : State Value} {events : List Event}
    (run : Run m s events t) (inv : Invariant m ref s) : Invariant m ref t := by
  induction run with
  | nil => exact inv
  | cons step run ih => exact ih (step_preserves reference inv step)

theorem reachable_invariant {Value : Type} [Inhabited Value]
    {m : Model Value} {ref : Field → Value} (reference : Reference m ref)
    {s : State Value} {events : List Event}
    (run : Run m initial events s) : Invariant m ref s :=
  run_preserves reference run (initial_invariant m ref)

theorem argument_agreement {Value : Type} [Inhabited Value]
    {m : Model Value} {ref : Field → Value} (reference : Reference m ref)
    {s : State Value} {events : List Event} (run : Run m initial events s)
    {c f : Nat} (started : c ∈ s.started) (read : f ∈ m.footprint c) :
    s.arguments c f = ref f :=
  (reachable_invariant reference run).arguments c started f read

theorem at_most_once {Value : Type} [Inhabited Value]
    {m : Model Value} {ref : Field → Value} (reference : Reference m ref)
    {s : State Value} {events : List Event} (run : Run m initial events s) :
    s.started.Nodup := (reachable_invariant reference run).unique

theorem capacity_safety {Value : Type} [Inhabited Value]
    {m : Model Value} {ref : Field → Value} (reference : Reference m ref)
    {s : State Value} {events : List Event} (run : Run m initial events s) :
    ∀ r, load m s.active r ≤ m.capacity r :=
  (reachable_invariant reference run).bounded

theorem effect_order {Value : Type} [Inhabited Value]
    {m : Model Value} {s t : State Value} {c : Call}
    (step : Step m s (.dispatch c) t) :
    ∀ p ∈ m.effectBefore c, p ∈ s.finished := by
  cases step with
  | dispatch c fresh ready early effects fits => exact effects

theorem publication_immutable {Value : Type} [Inhabited Value]
    {m : Model Value} {s t : State Value} {e : Event} {f : Field} {v : Value}
    (step : Step m s e t) (published : s.store f = some v) :
    t.store f = some v := by
  cases step with
  | dispatch => exact published
  | publish g running started declared unpublished =>
    have ne : f ≠ g := by
      intro h; subst f; rw [unpublished] at published; contradiction
    simpa only [put_other _ _ _ _ ne] using published
  | complete => exact published

theorem early_dispatch_requires_label {Value : Type} [Inhabited Value]
    {m : Model Value} {s t : State Value} {c : Call} {f : Field}
    (step : Step m s (.dispatch c) t) (read : f ∈ m.footprint c)
    (unfinished : m.owner f ∉ s.finished) : m.earlySafe c := by
  cases step with
  | dispatch c fresh ready early effects fits =>
    exact (early f read).resolve_left unfinished

theorem run_publication_immutable {Value : Type} [Inhabited Value]
    {m : Model Value} {s t : State Value} {events : List Event} {f : Field} {v : Value}
    (run : Run m s events t) (published : s.store f = some v) :
    t.store f = some v := by
  induction run with
  | nil => exact published
  | cons step run ih => exact ih (publication_immutable step published)

/-- Any two successful schedules agree on all observed fields. This does not
    assert equality of wall-clock times or of unconstrained external side effects. -/
theorem field_observational_equivalence {Value : Type} [Inhabited Value]
    {m : Model Value} {ref : Field → Value} (reference : Reference m ref)
    {s t : State Value} {left right : List Event}
    (a : Run m initial left s) (b : Run m initial right t)
    {f : Field} {x y : Value} (hx : s.store f = some x) (hy : t.store f = some y) :
    x = y := by
  have h1 := (reachable_invariant reference a).values f x hx
  have h2 := (reachable_invariant reference b).values f y hy
  exact h1.trans h2.symm

#print axioms reachable_invariant
#print axioms argument_agreement
#print axioms at_most_once
#print axioms capacity_safety
#print axioms effect_order
#print axioms publication_immutable
#print axioms early_dispatch_requires_label
#print axioms run_publication_immutable
#print axioms field_observational_equivalence

end CompactFlow
