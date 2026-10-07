import CompactFlow

/-! Non-vacuous early-dispatch witness and boundary check. -/
namespace CompactFlow.Examples

def model : Model Nat where
  footprint c := if c = 0 then [] else [0]
  owner f := f
  produces c := [c]
  eval c a := fun _ => if c = 0 then 7 else a 0
  exact := by
    intro c a b h
    funext f
    by_cases hc : c = 0
    · simp [hc]
    · have hv := h 0 (by simp [hc]); simp [hc, hv]
  earlySafe _ := True
  effectBefore _ := []
  demand _ r := if r = 0 then 1 else 0
  capacity r := if r = 0 then 2 else 0

theorem reference : Reference model (fun _ => 7) := by
  intro c f hf; simp [model]

def ds (s : State Nat) (c : Call) : State Nat :=
  { s with arguments := put s.arguments c (view s), result := put s.result c (model.eval c (view s)),
           started := c :: s.started, active := c :: s.active }
def ps (s : State Nat) (f : Field) : State Nat :=
  { s with store := put s.store f (some (s.result (model.owner f) f)) }
def cs (s : State Nat) (c : Call) : State Nat :=
  { s with active := remove c s.active, finished := c :: s.finished }
def s1 := ds initial 0
def s2 := ps s1 0
def s3 := ds s2 1
def s4 := ps s3 1
def s5 := cs s4 0
def s6 := cs s5 1

theorem d0 : Step model initial (.dispatch 0) s1 := by
  apply Step.dispatch
  · simp [initial]
  · simp [model]
  · simp [model]
  · simp [model]
  · intro r; by_cases h : r = 0 <;> simp [initial, load, model, h]
theorem p0 : Step model s1 (.publish 0) s2 := by
  apply Step.publish
  · simp [s1, ds, initial, model]
  · simp [s1, ds, initial, model]
  · simp [model]
  · simp [s1, ds, initial]
theorem d1 : Step model s2 (.dispatch 1) s3 := by
  apply Step.dispatch
  · simp [s2, ps, s1, ds, initial]
  · intro f hf
    have h : f = 0 := by simpa [model] using hf
    subst f
    exact ⟨7, by simp [s2, ps, s1, ds, model, put]⟩
  · intro f hf; exact Or.inr trivial
  · simp [model]
  · intro r; by_cases h : r = 0
    all_goals simp [s2, ps, s1, ds, initial, load, model, h]
theorem p1 : Step model s3 (.publish 1) s4 := by
  apply Step.publish
  · simp [s3, ds, model]
  · simp [s3, ds, model]
  · simp [model]
  · simp [s3, ds, s2, ps, s1, initial, put]
theorem c0 : Step model s4 (.complete 0) s5 := by
  apply Step.complete
  · simp [s4, ps, s3, ds, s2, s1, initial]
  · intro f hf
    have h : f = 0 := by simpa [model] using hf
    subst f
    exact ⟨7, by simp [s4, ps, s3, ds, s2, s1, model, put]⟩
theorem c1 : Step model s5 (.complete 1) s6 := by
  apply Step.complete
  · simp [s5, cs, s4, ps, s3, ds, s2, s1, initial, remove]
  · intro f hf
    have h : f = 1 := by simpa [model] using hf
    subst f
    exact ⟨7, by simp [s5, cs, s4, ps, s3, ds, s2, s1, model, put, view]⟩
theorem early_run : Run model initial
    [.dispatch 0, .publish 0, .dispatch 1, .publish 1, .complete 0, .complete 1] s6 :=
  .cons d0 (.cons p0 (.cons d1 (.cons p1 (.cons c0 (.cons c1 (.nil _))))))
/-- Consumer 1 starts with producer 0 still active, before its completion. -/
example : 0 ∈ s3.active ∧ 1 ∈ s3.started ∧ 0 ∉ s3.finished := by
  simp [s3, ds, s2, ps, s1, initial]
example : Invariant model (fun _ => 7) s6 := reachable_invariant reference early_run
example : s6.store 1 = some 7 := by
  simp [s6, cs, s5, s4, ps, s3, ds, s2, s1, model, put, view]
/-- An actually read but omitted field cannot satisfy the exactness premise. -/
theorem omitted_dependency_not_exact :
    ¬ (∀ a b : Nat → Nat, (∀ f ∈ ([] : List Nat), a f = b f) → a 0 = b 0) := by
  intro h
  have bad := h (fun _ => 0) (fun _ => 1) (by simp)
  contradiction
#print axioms early_run
#print axioms omitted_dependency_not_exact
end CompactFlow.Examples
