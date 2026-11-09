# VSAs.jl

Implementation of Vector Symbolic Architectures in Julia!

# Currently supported

Currently, the only implemented VSA are Holographic Reduced Representations.
More will come soon.

# Development

All commands below are run from the repository root. The package requires
Julia 1.11 or newer.

## Building the docs

The first time, or after `docs/Project.toml` changes, install the docs
dependencies:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
```

Then build:

```sh
julia --project=docs docs/make.jl
```

The site is written to `docs/build/` (ignored by git). Open
`docs/build/index.html` in a browser. The docs environment uses the local
source, so there is no need to reinstall the package after editing it.

The build runs every doctest and checks every `@ref` link. It fails if
either one is broken. `@ref` can only link to names that are documented in
this package, so refer to Base functions such as `argmax` with plain code
formatting instead.

## Running the tests

The first time, or after `test/Project.toml` changes:

```sh
julia --project=test -e 'using Pkg; Pkg.instantiate()'
```

Then run the whole suite:

```sh
julia --project=test test/runtests.jl
```

`julia --project -e 'using Pkg; Pkg.test()'` runs the same suite.

The suite contains:

| Test set | What it checks |
| --- | --- |
| Doctests | Every `jldoctest` block in a docstring produces its stated output |
| HRR element types | `HRR{Float32}` and `HRR{Float64}` keep their element type through every operation |
| Aqua | Package hygiene: method ambiguities, type piracy, undefined exports, missing compat entries |
| JET | Static analysis for errors such as calls with no matching method |

## Adding tests

When you add or change functionality:

1. **Give every public function a `jldoctest` example in its docstring.**
   It runs as part of the test suite with no further setup. Before each
   example, `using VSAs, Distributions, Random` has already run. Names that
   are `public` but not exported must be written in full, for example
   `VSAs.bind` or `VSAs.circular_convolution`.
2. **Add a `@testset` in `test/runtests.jl`** for anything a short example
   doesn't cover well, such as element-type preservation, thrown errors
   (`@test_throws`) or statistical properties with a fixed seed
   (`Xoshiro(42)`). Use a fixed RNG so failures can be reproduced.
3. **Add any new test-only package** to the test environment:

   ```sh
   julia --project=test -e 'using Pkg; Pkg.add("PackageName")'
   ```

4. **Run the full suite before committing.** Aqua and JET often catch
   problems in new code even when you didn't write tests for it, such as
   an exported name that was never defined.

If an intended change alters a doctest's output, rerun with
`doctest(VSAs; fix=true)` in place of `doctest(VSAs; manual=false)` once,
check the diff, then restore the original line.

## Adding a new VSA model

1. Create `src/models/<model>.jl` and `include` it at the end of
   `src/VSAs.jl`, then `export` the model type there.
2. Define `struct MyModel{T} <: AbstractVSA{T}` with the vector stored in a
   field called `data`. This gives you `length`, `==` and `show` from
   `src/generics.jl`.
3. Implement the required methods listed in the `AbstractVSA` docstring:
   `bind`, `unbind`, `bundle`, `similarity` and `permute`. Optionally add
   `unbundle`.
4. Implement sampling with the RNG as the first argument:
   `normal(rng::AbstractRNG, ::Type{MyModel{T}}, size::Integer)`, and
   `unitary` if the model has unitary vectors. The forms without an RNG,
   the vector forms of `bind`/`bundle`/`unbind`, `unpermute` and
   `simplecleanup` all come from `src/generics.jl`.
5. Add a test set like `HRR element types` for the new model.

# Style

The general style guide we follow is the
[JuliaReach guidelines](https://juliareach.github.io/JuliaReachDevDocs/latest/guidelines/).
On top of that, this project uses the following conventions.

## File layout

| File | Contents |
| --- | --- |
| `src/VSAs.jl` | `using` statements, `include`s, and all `export`/`public` lines |
| `src/interface.jl` | `AbstractVSA`, the generic function declarations with their full docstrings, and the `⊗ ⨸ ⊕ ⊖` aliases |
| `src/utils.jl` | Plain-vector helpers such as `circular_convolution` |
| `src/generics.jl` | Methods that work for any `AbstractVSA` |
| `src/models/*.jl` | One file per model: its type, constructors and methods |

Group related code within a file under section comments such as
`#= HRR Binding =#`.

## Exports

- Keep every `export` and `public` line in `src/VSAs.jl`, next to the
  `include` of the file that defines the name.
- Export the model types, the operators and names that do not clash with
  Base.
- Names that clash with Base, such as `bind`, are marked `public` instead.
  Users reach them as `VSAs.bind` or through the operators.
- Internal helpers are neither exported nor marked `public`.

## Functions

- Use the long `function ... end` form with a return type annotation for
  model methods. One-line generic fallbacks in `src/generics.jl` can use
  the short form.
- Check that the two vectors have the same length and throw
  `DimensionMismatch` if they don't. Throw `ArgumentError` for invalid
  arguments such as an empty data vector.
- Keep the element type: a `Float32` input should give a `Float32`
  output. Watch for `Float64` literals such as `1e-8`, which promote the
  result. Convert them with `T(1e-8)`.
- Random sampling functions take the RNG first and the model type second,
  following Base's `rand([rng,] T, dims)`.

## Docstrings

Every public function and type gets a docstring in this shape. Leave out
any section that doesn't apply:

````julia
"""
    bundle(x::HRR, y::HRR)::HRR

One-line summary of what the function does.

### Inputs

- `x, y` -- HRR vectors that share the same dimension.

### Output

What is returned.

### Throws

Throws `DimensionMismatch` when `x` and `y` have different lengths.

### Notes

```jldoctest
julia> rng = Xoshiro(123);

julia> (normal(rng, HRR, 10) ⊕ normal(rng, HRR, 10)) isa HRR
true
```
"""
````

- The full mathematical definition of an operation lives on its generic
  declaration in `src/interface.jl`. Model docstrings describe only how
  that model implements it.
- Write LaTeX between double backticks, escaping backslashes:
  ``` ``\\mathbf{x} \\otimes \\mathbf{y}`` ```.
- Make doctests deterministic: use a seeded RNG and test properties
  (`isa`, `==`, comparisons) rather than printing random numbers.
