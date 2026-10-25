# MLIR Primer

A running log of experiments, concepts, and commands learned while exploring MLIR.

**Published copy:** [github.com/ZiaulChoudhury/MLIR_for_dummies](https://github.com/ZiaulChoudhury/MLIR_for_dummies) (`mlir_primer.md` in that repo). Edit here in `llvm-src` when you are in the LLVM tree; copy and push to the GitHub repo when you want the public doc updated.

**How this doc grows:** After each exploration question, the answer is added in the section where it belongs (not a separate FAQ), so the narrative stays in order.

**Build location:** `build/bin/`  
**LLVM/MLIR version:** 21.1.8 (Release)  
**Target:** arm64-apple-darwin21.6.0

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Concept: MLIR Structure](#2-concept-mlir-structure)
3. [Experiment 1: Parse-and-Print Round Trip](#3-experiment-1-parse-and-print-round-trip)
4. [Experiment 2: The Canonicalize Pass](#4-experiment-2-the-canonicalize-pass)
5. [Concept: Dialects](#5-concept-dialects)
6. [Concept: Mixing Dialects](#6-concept-mixing-dialects)
7. [Experiment 3: Lowering to MemRef](#7-experiment-3-lowering-to-memref)
8. [Experiment 4: Lowering to LLVM IR](#8-experiment-4-lowering-to-llvm-ir)
9. [Experiment 5: Writing a Custom Pass](#9-experiment-5-writing-a-custom-pass)
10. [Experiment 6: A One-Operation Dialect](#10-experiment-6-a-one-operation-dialect)
11. [Guide: The](#11-guide-the-tadd-lowering-pass) `tadd` [Lowering Pass](#11-guide-the-tadd-lowering-pass)
12. [Command Reference](#12-command-reference)
13. [Files Created](#13-files-created)

---

## 1. Environment Setup

Verified that Clang and MLIR are built and available in `build/`.

| Tool             | Path                       | Status                                       |
| ---------------- | -------------------------- | -------------------------------------------- |
| `mlir-opt`       | `build/bin/mlir-opt`       | Working                                      |
| `mlir-translate` | `build/bin/mlir-translate` | Working                                      |
| `mlir-tblgen`    | `build/bin/mlir-tblgen`    | Working                                      |
| `clang`          | `build/bin/clang`          | Working (linking needs `-isysroot` on macOS) |

**macOS linking tip:** Set SDK before compiling with clang:

```bash
export SDKROOT=$(xcrun --show-sdk-path)
```

**Enabled CMake projects:** `clang;lld;mlir`

---

## 2. Concept: MLIR Structure

MLIR organizes code as a **deeply nested, hierarchical tree** of structural components. The core design principle is that **everything is an Operation**, and nesting is achieved by placing **Regions** inside those operations, which in turn contain **Blocks**, which contain more **Operations**.

### The hierarchical tree

```text
Operation (e.g., builtin.module)
  └── Region
        └── Block
              ├── Operation (e.g., func.func)
              │     └── Region
              │           └── Block
              │                 ├── Operation (e.g., scf.if)
              │                 │     ├── Region (Then Branch)
              │                 │     │     └── Block (Implicit/Explicit)
              │                 │     │           └── Operation (arith.addi)
              │                 │     └── Region (Else Branch)
              │                 │           └── Block (Implicit/Explicit)
              │                 │                 └── Operation (scf.yield)
              │                 └── Operation (func.return)
```

### Structural component breakdown

#### Operations (Ops)

An **Operation** is the fundamental unit of execution, semantics, and syntax in MLIR. Everything from a high-level function definition to a low-level machine instruction is an operation.

- **Traits:** Ops have a unique name (e.g., `arith.addi`), inputs (operands), outputs (results), and attributes (compile-time constants).
- **Nesting:** Ops are the components that can optionally encapsulate **one or more Regions** to create nested structural logic.

In `simple.mlir`, `arith.constant`, `arith.addi`, and `func.func` are all operations.

#### Regions

A **Region** is a structural container nested inside an Operation that holds **Blocks**.

- **Traits:** Regions do not possess their own properties like attributes or types; they simply define a localized structural fence or visibility scope.
- **Constraints:** A region **cannot** hold operations directly. It must hold blocks, which act as the immediate container for operations.

#### Blocks

A **Block** is a linear sequence of **Operations** that executes sequentially from top to bottom.

- **Control flow:** A block represents a basic block in compiler terminology. It must end with a **terminator operation** (like `scf.yield`, `cf.br`, or `func.return`) that transfers control or exits the region.
- **Arguments:** Blocks accept explicit **block arguments** (e.g., `^bb1(%val: i32)`). This design choice eliminates the need for traditional LLVM-style phi nodes.

#### Modules

A **Module** (`builtin.module`, written as `module { }`) acts as the top-level translation unit container.

- **Traits:** It is not a distinct structural class in core MLIR data structures; it is simply a **specialized, top-level Operation** containing a single region and a single block to hold all other high-level operations.

### Structural matrix

| Component | Immediate Parent | What it Contains | Primary Purpose | SSA Scope Hierarchy |
| :--- | :--- | :--- | :--- | :--- |
| **Module** | *None (top level)* | One Region | Top-level translation unit wrapper | Global / module-level |
| **Operation** | Block | 0 or more Regions | Represents a distinct semantic action | Defines new SSA values |
| **Region** | Operation | 1 or more Blocks | Provides hierarchical nesting/scoping | Introduces localized visibility |
| **Block** | Region | 0 or more Operations | Linear, sequential basic block execution | Defines block-level input arguments |

### Concrete example: demystifying the hierarchy

When writing an operation like an `scf.if` statement, MLIR allows you to skip naming blocks for cleaner syntax (syntactic sugar). Under the hood, the blocks are explicitly present.

Here is the fully expanded, explicit representation of a conditional statement illustrating all four structural layers:

```mlir
// LAYER 1: MODULE (a specialized top-level Operation)
builtin.module {

  // LAYER 2: OPERATION (func.func)
  // This operation defines a function and contains exactly ONE region.
  func.func @toy_conditional(%cond: i1, %a: i32, %b: i32) -> i32 {

    // LAYER 3: REGION (the function body scope)
    // LAYER 4: BLOCK (^entry)
    ^entry:

      // LAYER 2: OPERATION (scf.if)
      // This single operation holds TWO distinct internal regions.
      %result = scf.if %cond -> (i32) {

        // LAYER 3: REGION (the "then" region)
        ^then_block:  // LAYER 4: BLOCK
          // LAYER 2: OPERATION (arith.addi)
          %sum = arith.addi %a, %b : i32

          // TERMINATOR OPERATION (ends the block, yields value from region)
          scf.yield %sum : i32

      } else {

        // LAYER 3: REGION (the "else" region)
        ^else_block:  // LAYER 4: BLOCK
          // TERMINATOR OPERATION
          scf.yield %a : i32
      }

      // Back inside the ^entry block execution sequence...
      // LAYER 2: OPERATION (func.return — a terminator exiting the function region)
      func.return %result : i32
  }
}
```

### Key contrasts with LLVM IR

1. **Recursive vs. flat:** LLVM IR enforces a rigid, flat, 3-layer hierarchy (`Module -> Function -> BasicBlock -> Instruction`). An LLVM instruction cannot contain other blocks or instructions. MLIR uses a recursive tree structure (`Operation -> Region -> Block -> Operation`), allowing complex high-level operations (like loops and conditionals) to encapsulate their own control flow natively.
2. **Block arguments vs. phi nodes:** LLVM uses phi nodes at the beginning of basic blocks to merge incoming data paths. MLIR bypasses this by allowing blocks to accept input values directly as arguments, passing data forward similarly to a standard function call interface.

---

## 3. Experiment 1: Parse-and-Print Round Trip

**Goal:** Write a simple `.mlir` file and run `mlir-opt` to read it and print it back.

**File:** `simple.mlir`

```mlir
// A minimal MLIR module: add two constants and return the result.
module {
  func.func @main() -> i32 {
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %sum = arith.addi %one, %two : i32
    return %sum : i32
  }
}
```

**Command:**

```bash
build/bin/mlir-opt simple.mlir
```

**Output:** Same program, but SSA names are renamed by the pretty-printer:

```mlir
module {
  func.func @main() -> i32 {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.addi %c1_i32, %c2_i32 : i32
    return %0 : i32
  }
}
```

**Save output to a file:**

```bash
build/bin/mlir-opt simple.mlir -o simple.out.mlir
```

**Key insight:** Running `mlir-opt` with no passes only parses and pretty-prints. A second run on the output produces identical IR (stable fixpoint).

---

## 4. Experiment 2: The Canonicalize Pass

**Goal:** Understand what `-canonicalize` does.

**Command:**

```bash
build/bin/mlir-opt simple.mlir -canonicalize
```

**Before:**

```mlir
func.func @main() -> i32 {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = arith.addi %c1_i32, %c2_i32 : i32
  return %0 : i32
}
```

**After:**

```mlir
func.func @main() -> i32 {
  %c3_i32 = arith.constant 3 : i32
  return %c3_i32 : i32
}
```

**What canonicalize does:**

- Applies canonicalization patterns from all loaded dialects iteratively until a fixpoint
- **Constant folding** — `1 + 2` became `3`
- **Dead code elimination** — removed unused intermediate values
- **Algebraic simplification** — e.g. `x + 0` → `x`
- **Control-flow cleanup** — simplifies regions, dead args, etc.

| Command                            | Behavior                                 |
| ---------------------------------- | ---------------------------------------- |
| `mlir-opt file.mlir`               | Parse → pretty-print (no transformation) |
| `mlir-opt file.mlir -canonicalize` | Parse → simplify/normalize → print       |

Canonicalize is best-effort; it does not guarantee a fully canonical form.

---

## 5. Concept: Dialects

Dialects are namespaces for operations (`dialect.opname`). Each operation (see [Section 2](#2-concept-mlir-structure)) belongs to exactly one dialect. In `simple.mlir`:

| Dialect   | Operations                     | Role                             |
| --------- | ------------------------------ | -------------------------------- |
| `builtin` | `module { }`                   | Root container (no prefix)       |
| `func`    | `func.func`, `return`          | Function definition and return   |
| `arith`   | `arith.constant`, `arith.addi` | Integer constants and arithmetic |

The two explicitly named dialects in `simple.mlir` are `func` and `arith`.

---

## 6. Concept: Mixing Dialects

**Yes — you can mix dialect operations in a single** `.mlir` **file.** That is a core MLIR design point.

A single `module` can contain ops from many dialects as long as types and SSA values line up:

```mlir
module {
  func.func @example(%len: index) -> i32 {
    %c0 = arith.constant 0 : index
    %buf = memref.alloca(%len) : memref<?xi32>
    %v = memref.load %buf[%c0] : memref<?xi32>
    scf.for %i = %c0 to %len step %c1 iter_args(%acc = %v) -> (i32) {
      ...
      scf.yield %acc : i32
    }
    return %0 : i32
  }
}
```

**Rules of thumb:**

1. Types must match across dialect boundaries
2. Higher-level dialects (tensor, linalg) coexist with lower-level ones (memref, llvm) during lowering
3. Passes transform between dialect mixes (`-canonicalize`, `-convert-scf-to-cf`, etc.)
4. Custom dialects must be linked into your `mlir-opt` build

---

## 7. Experiment 3: Lowering to MemRef

**Goal:** Lower `simple.mlir` to the memref dialect.

**Key insight:** Scalar `i32` arithmetic **cannot** be lowered to memref. The memref dialect models **memory buffers** (arrays), not register-level scalars.

```bash
# Does nothing useful on scalar IR:
build/bin/mlir-opt simple.mlir -one-shot-bufferize="bufferize-function-boundaries"
```

### Option A: Write memref ops directly

**File:** `simple_memref.mlir`

```mlir
module {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32

    %buf = memref.alloca() : memref<2xi32>
    memref.store %one, %buf[%c0] : memref<2xi32>
    %c1 = arith.constant 1 : index
    memref.store %two, %buf[%c1] : memref<2xi32>

    %a = memref.load %buf[%c0] : memref<2xi32>
    %b = memref.load %buf[%c1] : memref<2xi32>
    %sum = arith.addi %a, %b : i32
    return %sum : i32
  }
}
```

```bash
build/bin/mlir-opt simple_memref.mlir
```

### Option B: Tensor/linalg → bufferize → memref

**File:** `simple_tensor.mlir`

```mlir
module {
  func.func @main() -> tensor<2xi32> {
    %c1 = arith.constant 1 : i32
    %init = tensor.empty() : tensor<2xi32>
    %filled = linalg.fill ins(%c1 : i32) outs(%init : tensor<2xi32>) -> tensor<2xi32>
    return %filled : tensor<2xi32>
  }
}
```

```bash
build/bin/mlir-opt simple_tensor.mlir \
  -one-shot-bufferize="bufferize-function-boundaries"
```

**Output:**

```mlir
func.func @main() -> memref<2xi32> {
  %c1_i32 = arith.constant 1 : i32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xi32>
  linalg.fill ins(%c1_i32 : i32) outs(%alloc : memref<2xi32>)
  return %alloc : memref<2xi32>
}
```

### Typical lowering pipeline (reference)

```
tensor / linalg  →  -one-shot-bufferize  →  memref
memref + loops   →  -convert-linalg-to-loops  →  memref + scf/cf
memref           →  -finalize-memref-to-llvm  →  llvm
```

---

## 8. Experiment 4: Lowering to LLVM IR

**Goal:** Lower `simple.mlir` all the way to LLVM IR text (`.ll`).

This is a **two-step** process.

### Step 1: MLIR → LLVM dialect (still MLIR)

```bash
build/bin/mlir-opt simple.mlir \
  -convert-arith-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts
```

**Output:**

```mlir
module {
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(3 : i32) : i32
    llvm.return %2 : i32
  }
}
```

| Pass                          | What it does                                         |
| ----------------------------- | ---------------------------------------------------- |
| `-convert-arith-to-llvm`      | `arith.*` → `llvm.*`                                 |
| `-convert-func-to-llvm`       | `func.func` / `return` → `llvm.func` / `llvm.return` |
| `-reconcile-unrealized-casts` | Cleans up cast ops left over from conversion         |

### Step 2: LLVM dialect → LLVM IR text

```bash
build/bin/mlir-opt simple.mlir \
  -convert-arith-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  | build/bin/mlir-translate -mlir-to-llvmir
```

**Output:**

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i32 @main() {
  ret i32 3
}
```

### Save to file

```bash
build/bin/mlir-opt simple.mlir \
  -convert-arith-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  | build/bin/mlir-translate -mlir-to-llvmir \
  -o simple.ll
```

### Pipeline form (equivalent)

```bash
build/bin/mlir-opt simple.mlir \
  -pass-pipeline="builtin.module(func.func(convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts)" \
  | build/bin/mlir-translate -mlir-to-llvmir
```

### Full lowering chain (for complex IR)

```
high-level (tensor, linalg)
  → -one-shot-bufferize
memref + scf
  → -convert-linalg-to-loops, -convert-scf-to-cf
arith, cf, memref
  → -convert-arith-to-llvm, -finalize-memref-to-llvm, -convert-cf-to-llvm, -convert-func-to-llvm
llvm dialect
  → mlir-translate -mlir-to-llvmir
LLVM IR (.ll)
```

---

## 9. Experiment 5: Writing a Custom Pass

**Goal:** Write a bare-bones MLIR pass named `list-dialects` that prints every dialect namespace used by operations in a `.mlir` file.

Read this section top to bottom: **what a pass is** → **files and outputs** → **code walkthrough** → **questions asked**.

---

### 9.1 What is an MLIR pass?

An **MLIR pass** is a unit of work the **pass manager** runs over **MLIR IR** (a tree of **operations** in SSA form). Each pass is a C++ class with a `runOnOperation()` method. The manager decides *when* to run it; your code decides *what* to do to (or with) the IR.

| Idea | Meaning |
|------|---------|
| **Input** | An in-memory `ModuleOp` (and everything nested under it) parsed from a `.mlir` file |
| **Anchor op** | The pass is typed on one op kind (here `module`). `getOperation()` returns that instance |
| **Transform pass** | Rewrites IR (replace ops, erase ops, insert ops) — e.g. lowering `tadd.add` in §11 |
| **Analysis pass** | Inspects IR and reports; may not change the module — **`list-dialects` is analysis** |

Passes are invoked from a **driver** (`mlir-opt`, `mlir-playground-opt`, `tadd-opt`) via the CLI flag **`-<pass-argument>`** (here `-list-dialects`). `MlirOptMain` parses the file, builds a `PassManager`, runs the requested passes, then **prints the module again** unless you only emit side output (like our dialect list on `llvm::outs`).

**What does the pass “work on”?** Not on the text file directly — on the **operation graph**:

```text
module                          ← ListDialectsPass anchored here (ModuleOp)
  func.func @main               ← func dialect
    arith.constant              ← arith dialect
    arith.addi
    func.return
```

`list-dialects` runs **once** on the `module`, then **walks** all nested ops to see which **dialect** owns each op (`arith`, `func`, `builtin`, …).

---

### 9.2 Project layout (every file)

```
mlirplayground/
├── CMakeLists.txt                 # external project root (includes, subdirs)
├── lib/
│   ├── CMakeLists.txt             # builds pass library
│   └── ListDialectsPass.cpp       # pass implementation + registration
└── tools/
    ├── CMakeLists.txt             # builds driver executable
    └── mlir-playground-opt.cpp    # main(): register pass + MlirOptMain
```

| File | Role |
|------|------|
| `mlirplayground/CMakeLists.txt` | When nested in the LLVM tree, sets MLIR include paths and adds `lib/` + `tools/`. Standalone build uses `find_package(MLIR)`. |
| `lib/CMakeLists.txt` | Defines `MLIRPlaygroundPasses` from `ListDialectsPass.cpp`; links `MLIRIR`, `MLIRPass`, `MLIRSupport`. |
| `lib/ListDialectsPass.cpp` | Pass class, `runOnOperation()`, `registerListDialectsPass()`. |
| `tools/CMakeLists.txt` | Builds `mlir-playground-opt`, links `MLIROptLib`, `MLIRPlaygroundPasses`, and all dialect libs (for parsing). |
| `tools/mlir-playground-opt.cpp` | Registers the pass and dialects, calls `MlirOptMain`. |

**Input file for experiments:** `simple.mlir` at the repo root (scalar `func` + `arith`).

---

### 9.3 Build outputs (artifacts)

After configuring LLVM with `LLVM_EXTERNAL_PROJECTS=mlirplayground`:

| Output | Path (typical) | What it is |
|--------|----------------|------------|
| Pass library | `build/lib/libMLIRPlaygroundPasses.a` (or `.dylib`) | Object code for `ListDialectsPass` |
| Driver | `build/bin/mlir-playground-opt` | Executable: parse `.mlir`, run passes, print IR |

```bash
cmake -S llvm -B build \
  -DLLVM_EXTERNAL_PROJECTS=mlirplayground \
  -DLLVM_EXTERNAL_MLIRPLAYGROUND_SOURCE_DIR=$PWD/mlirplayground

ninja -C build mlir-playground-opt
```

No TableGen and no generated `.inc` files for this experiment — only C++ compile and link.

---

### 9.4 Runtime flow and every kind of output

```text
simple.mlir
  → mlir-playground-opt (argv: file, -list-dialects)
  → parse into ModuleOp (dialects must be registered)
  → PassManager runs ListDialectsPass on module
  → runOnOperation(): walk ops, print dialect list to stdout
  → MlirOptMain prints the (unchanged) module to stdout
```

**1. Pass side output** (analysis text on **stdout**, from `llvm::outs` inside the pass):

```bash
build/bin/mlir-playground-opt simple.mlir -list-dialects
```

```
Dialects used (3):
  - arith
  - builtin
  - func
```

(Order is **sorted** alphabetically in the current source.)

**2. MLIR IR output** (same run): after the pass, the driver still prints the **module** (unchanged IR). You will see the full `module { func.func @main ... }` after the dialect list unless you redirect or use `-o`.

**3. Help output** (no input file required for listing passes):

```bash
build/bin/mlir-playground-opt --help
```

Includes a line for `list-dialects` from `getDescription()`.

**4. No output file** unless you pass **`-o path.mlir`** — then the **printed IR** goes to that file; the dialect list still goes to stdout.

---

### 9.5 Pass code: `ListDialectsPass.cpp` line by line

Full source as in the tree:

```cpp
//===- ListDialectsPass.cpp - Print dialects used in a module -------------===//
//
// Skeleton MLIR pass: walks the IR and prints each dialect namespace that
// appears in at least one operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct ListDialectsPass
    : public PassWrapper<ListDialectsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ListDialectsPass)

  StringRef getArgument() const final { return "list-dialects"; }
  StringRef getDescription() const final {
    return "Print the names of all dialects used in the module.";
  }

  void runOnOperation() override {
    llvm::StringSet<> dialectNames;
    getOperation().walk([&](Operation *op) {
      dialectNames.insert(op->getDialect()->getNamespace());
    });

    llvm::outs() << "Dialects used (" << dialectNames.size() << "):\n";
    SmallVector<StringRef> sortedNames;
    for (const auto &entry : dialectNames)
      sortedNames.push_back(entry.getKey());
    llvm::sort(sortedNames);
    for (StringRef name : sortedNames)
      llvm::outs() << "  - " << name << "\n";
  }
};

} // namespace

namespace mlir {
void registerListDialectsPass() { PassRegistration<ListDialectsPass>(); }
} // namespace mlir
```

| Lines | What they do |
|-------|----------------|
| 8–12 | **Includes.** `BuiltinOps.h` → `ModuleOp`. `Pass.h` → `PassWrapper`, `OperationPass`, `PassRegistration`. LLVM headers for set, sort, `llvm::outs`. |
| 14 | `using namespace mlir` for brevity in the pass body. |
| 16 | Anonymous namespace — pass struct is file-local; only `registerListDialectsPass` is exported. |
| 18–19 | **Pass class** inherits `PassWrapper<ListDialectsPass, OperationPass<ModuleOp>>`: hand-written pass, anchored on **`module`**. |
| 20 | `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` — RTTI id for the pass (required by current MLIR pass infra). |
| 22 | `getArgument()` → CLI flag **`-list-dialects`** (no `list-dialects` prefix in the string). |
| 23–25 | `getDescription()` → text in `--help`. |
| 27–40 | **`runOnOperation()`** — body runs with `getOperation()` equal to the current `ModuleOp`. |
| 28 | `StringSet` stores unique dialect namespace strings. |
| 29–31 | **`.walk`**: preorder visit of every op under the module; insert `op->getDialect()->getNamespace()`. |
| 33 | Print header and count. |
| 34–37 | Copy set keys into `SmallVector` and **sort** for stable output. |
| 38–39 | Print one line per dialect. |
| 43 | Close anonymous namespace. |
| 45–47 | **`registerListDialectsPass`**: `PassRegistration<ListDialectsPass>()` registers the pass globally so `MlirOptMain` can find `-list-dialects`. Must be called from `main` before parsing. |

---

### 9.6 Driver: `mlir-playground-opt.cpp` line by line

```cpp
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir {
void registerListDialectsPass();
}

int main(int argc, char **argv) {
  mlir::registerListDialectsPass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR Playground optimizer driver\n", registry));
}
```

| Lines | What they do |
|-------|----------------|
| 7–8 | Pull in dialect registration and the standard opt driver entry point. |
| 10–12 | Forward declare our registration function (defined in the pass `.cpp` library). |
| 15 | Register **our** pass with the global pass registry. |
| 17–18 | Build a **dialect registry** and register **all** stock dialects so `simple.mlir` can parse (`arith`, `func`, …). |
| 20–21 | `MlirOptMain`: parse CLI, parse input MLIR, run passes, print IR. `asMainReturnCode` maps failure to an exit code. |

We do **not** call `registerAllPasses()` here — that pulls in hundreds of passes and **link errors** unless every conversion library is linked. We only register the one pass we need.

---

### 9.7 CMake (what each file adds)

**`lib/CMakeLists.txt`**

```cmake
add_mlir_library(MLIRPlaygroundPasses
  ListDialectsPass.cpp
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
)
```

Produces **`MLIRPlaygroundPasses`** containing the pass and `registerListDialectsPass`.

**`tools/CMakeLists.txt`**

```cmake
add_llvm_executable(mlir-playground-opt mlir-playground-opt.cpp)
target_link_libraries(mlir-playground-opt PRIVATE
  MLIROptLib
  MLIRPlaygroundPasses
  ${dialect_libs}
)
```

Links the driver to the pass library and dialect libs so parsing works.

---

### 9.8 Quick reference (pass anatomy)

| Piece | Role |
|-------|------|
| `PassWrapper<Derived, OperationPass<OpT>>` | Base class; `OpT` is the root op the pass runs on |
| `getArgument()` | Pass name on the CLI (e.g. `-list-dialects`) |
| `getDescription()` | Shown in `--help` |
| `runOnOperation()` | Your logic; `getOperation()` is the current root op |
| `getOperation().walk(...)` | Visit every nested operation under the anchored op |
| `PassRegistration<>` | Makes the pass discoverable by `MlirOptMain` |

---

### 9.9 Questions asked (from exploration)

**What is `PassWrapper`?**  
MLIR’s template base for a **hand-written** pass (no TableGen `Passes.td`). You subclass `PassWrapper<YourPass, OperationPass<...>>`, implement `getArgument()`, `getDescription()`, and `runOnOperation()`, and expose it with `PassRegistration<YourPass>()`.

**What is `OperationPass`?**  
A pass anchored on **one op type**. The pass manager calls `runOnOperation()` once per matching op in scope:

| Anchor | Runs |
|--------|------|
| `OperationPass<ModuleOp>` | Once per top-level `module` |
| `OperationPass<func::FuncOp>` | Once per `func.func` |

Nested ops are reached with `.walk()` (or explicit region iteration), not by choosing a different anchor for every op kind.

**Why `ModuleOp` for `list-dialects`?**  
A single-file example is one `module`; all ops nest under it. One module pass + `.walk()` matches “dialects used in this file.” A `func::FuncOp` pass would run per function — fine for per-function transforms, awkward for a whole-file inventory.

**Why not `registerAllPasses()` in the driver?**  
It registers every upstream pass but requires linking all corresponding libraries. For a minimal playground driver, register only `registerListDialectsPass()`.

**Does this pass change the IR?**  
No. It only reads via `walk` and prints. The printed MLIR after the run is the same module (aside from normal round-trip naming).

**Next steps for pass writing**

- TableGen `Passes.td` when you need pass options or pipeline YAML integration
- `OperationPass<func::FuncOp>` for per-function lowering (see `tadd-lower-to-linalg` in §11)
- `mlir/test/lib/IR/TestPrintDefUse.cpp` — more inspection patterns
- `mlir/examples/standalone/` — full out-of-tree dialect + pass project

---

## 10. Experiment 6: A One-Operation Dialect

**Goal:** A dialect with a single op that adds two `tensor<1x8xf32>` values (shape 1×8, eight `f32` elements).

The op is `tadd.add`. It checks types only. It does not compute the sum; that would be a later lowering or folder.

Clarifications from Q&A are merged into the subsections below (driver, `func`, type constraints, build order, etc.).

### Project

```
tadd/
├── include/TAdd/TAddDialect.td   # dialect name "tadd"
├── include/TAdd/TAddOps.td       # op tadd.add
├── lib/TAddDialect.cpp           # register the op
├── tools/tadd-opt.cpp            # parser driver
└── add.mlir                      # example
```

### Creating the dialect: every piece

A dialect is a **namespace of operations** (here `tadd`), not a new kind of IR container. `tadd.add` is still a normal MLIR operation: operands, result, type, traits. The work is to declare that op, generate its C++ class, register the dialect in a parser, and link a binary that knows about it.

The pieces, in the order they depend on each other:

| Step | Piece | File | What it produces |
|------|--------|------|------------------|
| 1 | Dialect declaration | `include/TAdd/TAddDialect.td` | Namespace `tadd`, C++ namespace `mlir::tadd` |
| 2 | Operation declaration | `include/TAdd/TAddOps.td` | Op `tadd.add`, its types, syntax, traits |
| 3 | TableGen | `include/TAdd/CMakeLists.txt` | Generated `*.inc` headers and method bodies |
| 4 | C++ headers | `TAddDialect.h`, `TAddOps.h` | Include the generated declarations |
| 5 | Registration | `lib/TAddDialect.cpp` | Dialect loads `AddOp`; op methods are compiled in |
| 6 | Library | `lib/CMakeLists.txt` | `libMLIRTAdd` |
| 7 | Driver | `tools/tadd-opt.cpp` | Binary that registers the dialect and parses a file |
| 8 | Link | `tools/CMakeLists.txt` | `tadd-opt` linked to `MLIRTAdd` and `func` |

This is dependency order, not the order you type files. You write the `.td` files, the headers, `TAddDialect.cpp`, the driver, and all three `CMakeLists.txt` first. At build time the order is: TableGen writes the `.inc` files, then `TAddDialect.cpp` and `tadd-opt.cpp` compile, then `libMLIRTAdd` links, then `tadd-opt` links. The driver does not wait on the library except at that final link.

#### 1. Declare the dialect (`TAddDialect.td`)

```tablegen
def TAdd_Dialect : Dialect {
  let name = "tadd";
  let cppNamespace = "::mlir::tadd";
}

class TAdd_Op<string mnemonic, list<Trait> traits = []> :
    Op<TAdd_Dialect, mnemonic, traits>;
```

`name = "tadd"` is the prefix in the text form (`tadd.add`). `cppNamespace` is where the generated C++ class lives (`mlir::tadd::TAddDialect`, `mlir::tadd::AddOp`). `TAdd_Op` is only a shortcut so every op in this dialect is written as `TAdd_Op<"mnemonic", [traits]>`.

#### 2. Declare the one operation (`TAddOps.td`)

Three parts.

**Type constraint.** `tensor<1x8xf32>` is a builtin type, not a `tadd` type. The op only accepts that shape:

- rank 2
- dimension 0 is 1
- dimension 1 is 8
- element type is `f32`

That is `TAdd_Tensor1x8xF32`. A `tensor<8xf32>` fails this check.

**Operands and result.**

```tablegen
def TAdd_AddOp : TAdd_Op<"add", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins TAdd_Tensor1x8xF32:$lhs, TAdd_Tensor1x8xF32:$rhs);
  let results = (outs TAdd_Tensor1x8xF32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}
```

`ins` **/** `outs` **and** `TAdd_Tensor1x8xF32` **— not** `tadd` **types**

| Name | What it is |
|------|------------|
| `ins` / `outs` | TableGen syntax for **operands** (inputs) and **results** (outputs) of the op |
| `TAdd_Tensor1x8xF32` | A **type constraint** defined only in `TAddOps.td` (`TypeConstraint` + `CPred`). It is a reusable name for a verifier predicate, not a new MLIR type |
| Actual MLIR type | Builtin `tensor<1x8xf32>` (`RankedTensorType` in the `builtin` dialect) |

The `TAdd_` prefix is naming convention in TableGen (this file’s constraints). It does **not** mean the dialect owns a type `!tadd.tensor<...>`. The `tadd` project generates empty `TAddOpsTypes.h.inc` / `.cpp.inc` because we never defined `TypeDef` entries.

In the printed IR you always see `tensor<1x8xf32>`, never `TAdd_Tensor1x8xF32`.

- Mnemonic `"add"` plus dialect name `tadd` gives the printed name `tadd.add`.
- `Pure` means no memory side effects (needs `SideEffectInterfaces.td`).
- `SameOperandsAndResultType` means both inputs and the result share one type (needs `InferTypeOpInterface.td`). The printer therefore writes the type once.
- `assemblyFormat` is the custom syntax. It expands to `%0 = tadd.add %a, %b : tensor<1x8xf32>`. `attr-dict` is the `{...}` attribute slot, empty in the example.

This op does not add numbers. It only records "this value is the sum of these two tensors." Computing the sum is a later pass or lowering.

#### 3. Generate C++ (`add_mlir_dialect`)

```cmake
add_mlir_dialect(TAddOps tadd)
```

TableGen reads `TAddOps.td` (which includes the dialect file) and writes, under `build/tools/tadd/include/TAdd/`:

| Generated file | Contents |
|----------------|----------|
| `TAddOpsDialect.h.inc` | `class TAddDialect`, declares `initialize()` |
| `TAddOpsDialect.cpp.inc` | Constructor; it calls `initialize()` |
| `TAddOps.h.inc` | `class AddOp` declaration |
| `TAddOps.cpp.inc` | Parse, print, verify, and effect methods. Included twice under different macros |
| `TAddOpsTypes.h.inc` / `.cpp.inc` | Empty here (no `TypeDef` in TableGen) |

You do not edit the `.inc` files. Change the `.td` and rebuild.

**Who runs** `mlir-tblgen`**?** You do not run it by hand in normal builds. `add_mlir_dialect` in `mlir/cmake/modules/AddMLIR.cmake` registers rules; Ninja runs them when building target `MLIRTAddOpsIncGen` (or anything that depends on it, e.g. `MLIRTAdd`). Example command from a generated script:

```bash
build/bin/mlir-tblgen -gen-op-defs ... tadd/include/TAdd/TAddOps.td -o TAddOps.cpp.inc
```

Outputs live under `build/tools/tadd/include/TAdd/`, not in the source tree.

**What is inside each** `.inc` **file (examples from** `tadd`**):**

| File | Role | Example of generated content |
|------|------|------------------------------|
| `TAddOpsDialect.h.inc` | Dialect class declaration | `class TAddDialect : public Dialect { ... getDialectNamespace() returns "tadd"; }` |
| `TAddOpsDialect.cpp.inc` | Dialect constructor | `TAddDialect::TAddDialect(...) { initialize(); }` |
| `TAddOps.h.inc` | Op C++ API (see macros below) | `class AddOp` with `getOperationName() == "tadd.add"`, `getLhs()`, `getRhs()`, `create(...)` |
| `TAddOps.cpp.inc` | Op list + implementations (see macros) | Type verifier for `tensor<1x8xf32>`, `AddOp::parse`, `AddOp::print`, `build`, trait hooks |
| `TAddOpsTypes.h.inc` | Custom dialect types | Empty for `tadd` (only `#ifdef GET_TYPEDEF_CLASSES` shell) |

Mapping one line of IR to generated code:

```mlir
%0 = tadd.add %a, %b : tensor<1x8xf32>
```

| IR piece | Generated hook |
|----------|----------------|
| Name `tadd.add` | `AddOp::getOperationName()` |
| Text syntax | `AddOp::parse` / `AddOp::print` (from `assemblyFormat` in `.td`) |
| Type `tensor<1x8xf32>` | Verifier from `TAdd_Tensor1x8xF32` (`CPred` in `.td`) |
| Op known to parser | `GET_OP_LIST` → `addOperations<AddOp>()` in `initialize()` |

`GET_OP_CLASSES` **/** `GET_OP_LIST` **— ODS convention (not C++ standard)**

These are **flags you** `#define` **before** `#include` a generated `*.inc` file. The file contains multiple `#ifdef` sections; the flag picks which slice is pasted in. For ops defined via `*Ops.td` and `add_mlir_dialect`, the same pattern is used across MLIR.

| Define before `#include` | File              | What you get                                        |
| ------------------------ | ----------------- | --------------------------------------------------- |
| *(none)*                 | `TAddOps.h.inc`   | Forward declaration only: `class AddOp;`            |
| `GET_OP_CLASSES`         | `TAddOps.h.inc`   | Full `AddOp` **class declaration**                  |
| `GET_OP_LIST`            | `TAddOps.cpp.inc` | `::mlir::tadd::AddOp` (for `addOperations<...>()`)  |
| `GET_OP_CLASSES`         | `TAddOps.cpp.inc` | `AddOp` **method bodies** (parse, print, verify, …) |

Custom dialect **types** use a different flag (`GET_TYPEDEF_CLASSES`) in `*Types.h.inc`. Hand-written ops without TableGen do not use these macros.

#### 4. Headers that pull the declarations in

`TAddDialect.h` includes `TAddOpsDialect.h.inc`, which is the generated dialect class.

`TAddOps.h` includes the interfaces the traits need (`InferTypeOpInterface.h`, `SideEffectInterfaces.h`), then:

```cpp
#define GET_OP_CLASSES
#include "TAdd/TAddOps.h.inc"
```

That pulls in the full `AddOp` declaration (not just `class AddOp;`).

#### 5. Register the op and compile its methods (`TAddDialect.cpp`)

The generated dialect constructor calls `TAddDialect::initialize()`, but it does not write that function. The hand-written file does, in two includes of the same `TAddOps.cpp.inc`:

```cpp
void TAddDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TAdd/TAddOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "TAdd/TAddOps.cpp.inc"
```

- `GET_OP_LIST` expands to the type list `AddOp`, so the dialect knows the op exists.
- `GET_OP_CLASSES` the second time expands to the **method bodies**: `parse`, `print`, `verifyInvariantsImpl`, `inferReturnTypes`, `getEffects`.

**What** `initialize()` **lines 10–15 do (after preprocessing):**

```cpp
void TAddDialect::initialize() {
  addOperations<::mlir::tadd::AddOp>();
}
```

`#define GET_OP_LIST` before `#include "TAdd/TAddOps.cpp.inc"` selects the small fragment in the generated file that is only `::mlir::tadd::AddOp`. `addOperations<...>()` registers that C++ op class with the dialect so the parser can create `tadd.add` when it sees the mnemonic. This runs once per dialect load, before any `.mlir` file is parsed.

`#include "mlir/IR/OpImplementation.h"` must come before those bodies. The generated parser uses `OpAsmParser`, which is only a forward declaration in the core headers.

Until `initialize()` runs, the parser has no `tadd.add`, even if the `.td` file exists.

#### 6. Library (`lib/CMakeLists.txt`)

`add_mlir_dialect_library(MLIRTAdd ...)` compiles `TAddDialect.cpp` into `libMLIRTAdd`.

- `DEPENDS MLIRTAddOpsIncGen` waits for TableGen. The target name is `MLIR` + the `TAddOps` argument of `add_mlir_dialect` + `IncGen`.
- `MLIRInferTypeOpInterface` and `MLIRSideEffectInterfaces` match the traits on the op.

#### 7. Driver (`tadd-opt.cpp`)

```cpp
registry.insert<mlir::tadd::TAddDialect, mlir::func::FuncDialect>();
return mlir::MlirOptMain(argc, argv, "...", registry);
```

`MlirOptMain` parses command-line arguments, parses the `.mlir` file with the dialects in `registry`, runs any passes (none here), and prints the module. `func` is in the registry because `add.mlir` uses `func.func`. `builtin` (`module`, `tensor`) is already part of MLIR and is not inserted here.

#### 8. What happens when you run it

```text
tadd/add.mlir
  → tadd-opt builds a DialectRegistry { tadd, func }
  → parser sees func.func          (func dialect)
  → parser sees tadd.add           (tadd dialect, registered in initialize())
  → verifier checks both operands and the result are tensor<1x8xf32>
  → printer writes the module back out
```

No pass runs. The printed IR is the same program with SSA names renamed (`%a` → `%arg0`).

### Example (`tadd/add.mlir`)

```mlir
func.func @main(%a: tensor<1x8xf32>, %b: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = tadd.add %a, %b : tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}
```

### Why `tadd-opt`, not `mlir-opt`?

`mlir-opt` only parses dialects that were compiled into that binary and registered in its `DialectRegistry`. `tadd` is a separate project, so stock `mlir-opt` does not know `tadd.add`.

`tadd-opt` is a tiny copy of that driver that inserts `TAddDialect` (and `func`) before parsing. To use `mlir-opt` instead, the dialect would have to be linked into `mlir-opt`, or loaded as a dialect plugin (`--load-dialect-plugin`).

### Why `func` as well as `tadd`?

`add.mlir` uses `func.func` and `return`. Those ops belong to the `func` dialect, so the parser must have it registered. A file that contained only `tadd` ops would not need `func`.

A one-line `%0 = tadd.add %a, %b` still cannot stand alone. `%a` and `%b` are SSA values, so something must define them. `func.func` does that by making them block arguments. `tadd.add` itself has no region and no constants.

The `builtin` dialect (`module`, `tensor`) is always loaded by MLIR. The driver does not register it.

### Lowering `add.mlir` to memref

See **[§11](#11-guide-the-tadd-lowering-pass)** for a step-by-step, beginner-friendly explanation of the `tadd-lower-to-linalg` pass and how it fits with bufferization.

Quick command:

```bash
build/bin/tadd-opt tadd/add.mlir \
  -tadd-lower-to-linalg \
  -one-shot-bufferize=bufferize-function-boundaries
```

Reference output: `tadd/add_memref.mlir`.

### Build and run

```bash
cmake -S llvm -B build \
  -DLLVM_EXTERNAL_PROJECTS="mlirplayground;tadd" \
  -DLLVM_EXTERNAL_TADD_SOURCE_DIR=$PWD/tadd

ninja -C build tadd-opt
build/bin/tadd-opt tadd/add.mlir
```

A different shape (for example `tensor<8xf32>`) is rejected by the type constraint on the op.

---

## 11. Guide: The `tadd` Lowering Pass

This section explains **what lowering means** for our tiny `tadd` dialect, how the pass `tadd-lower-to-linalg` works, and how that connects to **memref**. No prior MLIR compiler background required — think of it as translating your program into dialects that already know how to talk about memory.

### What problem are we solving?

In `tadd/add.mlir` you wrote something that *looks* like addition:

```mlir
%0 = tadd.add %a, %b : tensor<1x8xf32>
```

That line is **not** machine code. It is a **placeholder op** in the `tadd` dialect: it says “these two tensors are added,” but it does not say *how* (loops, memory, SIMD, etc.). That is fine for a teaching dialect.

To get closer to something a CPU could run, we move through **layers** of MLIR:

```text
tadd.add on tensor     →  linalg + arith on tensor  →  linalg on memref
   (your dialect)            (standard tensor ops)         (buffers in memory)
```

The first arrow is `tadd-lower-to-linalg`. The second is `-one-shot-bufferize` (covered in [§7](#7-experiment-3-lowering-to-memref); we only summarize it here).

### Key idea: “lowering” = replace ops with simpler, more standard ops

**Lowering** (in this doc) means: find ops in your dialect and **rewrite** them into ops from dialects MLIR already understands (`linalg`, `arith`, `tensor`, later `memref`).

You are not changing the *math* (still element-wise add of eight floats in a 1×8 grid). You are changing the **vocabulary** so the rest of the compiler pipeline can take over.

| Stage | What you see in the IR | Who understands it |
|-------|------------------------|--------------------|
| Start | `tadd.add` | Only `tadd-opt` with `TAddDialect` registered |
| After `tadd-lower-to-linalg` | `linalg.generic`, `arith.addf`, `tensor.empty` | Most tensor pipelines + bufferizer |
| After `one-shot-bufferize` | `memref.alloc`, `linalg.generic` on memref | Lowering toward loops / LLVM |

### Try it yourself: see each stage

From the repo root:

```bash
# Stage 0 — parse and print (no pass)
build/bin/tadd-opt tadd/add.mlir

# Stage 1 — only the lowering pass
build/bin/tadd-opt tadd/add.mlir -tadd-lower-to-linalg

# Stage 2 — lowering + bufferize → memref
build/bin/tadd-opt tadd/add.mlir \
  -tadd-lower-to-linalg \
  -one-shot-bufferize=bufferize-function-boundaries
```

Compare the middle output to the last: function arguments change from `tensor<...>` to `memref<...>`, and `tensor.empty` becomes `memref.alloc`. The **add** is still expressed as a `linalg.generic` with `arith.addf` inside — only the *containers* (tensor vs memref) changed.

### Before and after: one `tadd.add` line

**Before** (`tadd/add.mlir`) — one custom op:

```mlir
func.func @main(%a: tensor<1x8xf32>, %b: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = tadd.add %a, %b : tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}
```

**After** `tadd-lower-to-linalg` **only** — same function, but the add is spelled out:

```mlir
func.func @main(%arg0: tensor<1x8xf32>, %arg1: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = tensor.empty() : tensor<1x8xf32>
  %1 = linalg.generic { ... } ins(%arg0, %arg1 : ...) outs(%0 : ...) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.addf %in, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<1x8xf32>
  return %1 : tensor<1x8xf32>
}
```

Read that `linalg.generic` block in plain English:

1. `tensor.empty` — create an output tensor (values undefined until filled).
2. `linalg.generic` — for each index in the shape, read one element from `%arg0` and `%arg1`, add them, write into the output.
3. `arith.addf` — the actual `f32` add at one point.
4. `linalg.yield` — return that scalar from the inner block.

So the lowering pass **expands** `tadd.add` into “allocate output + loop nest spelled as linalg + float add.”

#### Why `linalg.generic` instead of something shorter?

`tadd.add` means **element-wise add** over a shaped tensor: for every index `(i, j)`, output[i,j] = lhs[i,j] + rhs[i,j]. In MLIR that is usually written as a **structured** linalg op, not as explicit `scf.for` loops in the first lowering step.

`linalg.generic` is the general form: you declare

- **iterator types** — here both dimensions are `"parallel"` (no dependencies between iterations),
- **indexing maps** — here identity maps so all three tensors use the same `(d0, d1)` indices,
- a small **region** — here one `arith.addf` and `linalg.yield`.

That is exactly “a nest of parallel loops; each iteration does one float add.” The bufferizer and later passes (`convert-linalg-to-loops`, fusion, etc.) already know how to handle `linalg.generic` on tensors and memrefs.

**Why not keep** `tadd.add` **and teach the bufferizer about it?** You could register custom bufferization for `tadd.add`, but for a learning project it is simpler to **reuse** linalg’s existing rules.

**Why not emit** `scf` **loops +** `memref.load`**/**`store` **immediately?** That skips the tensor/linalg layer. You would hand-write memory semantics early. The usual pipeline is: high-level or custom op → **linalg on tensor** → bufferize → lower linalg to loops.

**Named linalg ops** (e.g. generated `linalg.matmul`, `linalg.fill`) exist for common patterns. A plain tensor add can sometimes be expressed with a named op, but `linalg.generic` is the straightforward choice in a tiny rewrite: no extra TableGen, and the printed IR shows the indexing maps and the `arith.addf` in the body.

#### Can I use a standard linalg op (e.g. `linalg.add`) instead?

**Yes — for this kernel, you should think of** `linalg.add` **as the “standard” op.** MLIR generates it from `LinalgNamedStructuredOps.yaml` as a destination-style elementwise add. After lowering with `linalg.add` instead of `linalg.generic`, bufferization and later passes behave the same way.

Example shape of the rewrite (conceptually):

```mlir
%0 = tensor.empty() : tensor<1x8xf32>
%1 = linalg.add ins(%a, %b : tensor<1x8xf32>, tensor<1x8xf32>)
               outs(%0 : tensor<1x8xf32>) -> tensor<1x8xf32>
```

In C++, `linalg::AddOp::create(rewriter, loc, resultType, inputs, outputs)` (see generated `LinalgStructuredOps.h.inc`).

**Why the sample pass uses** `linalg.generic` **anyway:**

| Reason | Detail |
|--------|--------|
| Teaching | Generic makes indexing maps, parallel iterators, and the `arith.addf` region visible in one place |
| One pattern for many ops | The same rewrite template works for any elementwise op you invent in `tadd` without picking a named op |
| Not a hard rule | Production code often emits `linalg.add`, `linalg.map { arith.addf }`, or imports from another frontend that already uses named ops |

So: **you are not blocked from standard linalg ops** — `linalg.generic` here is a pedagogical and flexible choice, not a requirement of the bufferizer.

**After bufferize** — see `tadd/add_memref.mlir`: tensors at the function boundary become **memref** (memory buffers), and `tensor.empty` becomes `memref.alloc`.

### Why not bufferize `tadd.add` directly?

The bufferizer answers: “Given tensor ops, where should data live in memory?” It only knows ops that implement **bufferization interfaces** (built into `linalg`, `tensor.empty`, etc.).

`tadd.add` is custom. Nobody registered rules like “replace `tadd.add` with memref loads/stores.” So if you run:

```bash
build/bin/tadd-opt tadd/add.mlir -one-shot-bufferize=...
```

…the pass stops with an error: it does not know how to bufferize `tadd.add`.

**Fix:** lower to `linalg` first, then bufferize. That is the normal pattern for user-defined tensor dialects.

#### Once I’m on linalg, can I bufferize?

**Yes.** After `tadd-lower-to-linalg`, the function only uses tensor ops the one-shot bufferizer already understands (`tensor.empty`, `linalg.generic`, `arith.addf`, plus `func` boundaries). Run:

```bash
build/bin/tadd-opt tadd/add.mlir -tadd-lower-to-linalg -one-shot-bufferize=bufferize-function-boundaries
```

Or bufferize the intermediate file in two steps: `-tadd-lower-to-linalg -o /tmp/x.mlir`, then `-one-shot-bufferize=bufferize-function-boundaries` on `/tmp/x.mlir`.

**Caveats (still beginner-friendly):**

- IR must stay on **tensor** until bufferize finishes — you are not mixing memref and tensor in the same value chain without the bufferizer’s help.
- The **driver** must register bufferization interfaces for `tensor`, `linalg`, `arith`, and `func` (see `tadd-opt.cpp`). Stock `mlir-opt` does that by default; minimal `tadd-opt` needed the extra registration.
- Use `bufferize-function-boundaries` if you want `func` arguments and results to become memref (as in `tadd/add_memref.mlir`).

Linalg on **memref** (after bufferize) is the usual next step toward loops or LLVM — that is a different set of passes, not another bufferize.

### What the pass is, in MLIR terms

| Question | Answer for `tadd-lower-to-linalg` |
|----------|-----------------------------------|
| CLI name | `-tadd-lower-to-linalg` |
| Source file | `tadd/lib/LowerTAddToLinalg.cpp` |
| Runs on | Each `func.func` (`OperationPass<func::FuncOp>`) |
| Mechanism | **Greedy pattern rewrite** — find `tadd.add`, replace with new ops |
| Registers in | `registerTAddLowerToLinalgPass()` called from `tadd-opt.cpp` |

This is the same *family* of pass as in [§9](#9-experiment-5-writing-a-custom-pass): subclass `PassWrapper`, implement `runOnOperation()`, expose with `PassRegistration`. The difference is that instead of only walking and printing, we **rewrite** the IR with an `OpRewritePattern`.

### Inside the pass (conceptual, not every line of C++)

The pass does three mechanical things:

1. **Load dialects** it will create (`arith`, `linalg`, `tensor`) inside `runOnOperation()`.
2. **Register a pattern** `LowerTAddAddOp` that says: “when you see `tadd::AddOp`, run this rewrite.”
3. **Run** `applyPatternsGreedily` on the current function until no more `tadd.add` ops match.

The pattern body (simplified story):

```text
1. Create tensor.empty for the result shape (1×8, f32).
2. Build linalg.generic with parallel loops over both dimensions.
3. Inside the generic’s region: arith.addf on the two input scalars, linalg.yield the sum.
4. Replace the old tadd.add result with the generic’s result.
```

Identity **indexing maps** (`#map` in the printed IR) mean “input and output indices line up” — for this shape, it is a straightforward element-wise map.

If the op is not `f32` or not a ranked tensor, the pattern returns **failure** and leaves the op alone (our dialect only allows `tensor<1x8xf32>` anyway).

#### Is `matchAndRewrite` an MLIR override?

Yes. `LowerTAddAddOp` inherits from `OpRewritePattern<tadd::AddOp>`. The base class declares a **pure virtual** method:

```cpp
virtual LogicalResult matchAndRewrite(tadd::AddOp op,
                                      PatternRewriter &rewriter) const = 0;
```

Your `LowerTAddAddOp::matchAndRewrite` **overrides** that method. The driver never calls it directly: `applyPatternsGreedily` walks the function, and for each op that matches the pattern’s root type (`tadd::AddOp`), MLIR calls the base’s `final` `matchAndRewrite(Operation *, …)`, which `cast`s to `AddOp` and dispatches to your override.

| Return value | Meaning |
|--------------|---------|
| `success()` | Rewrite applied; old op is replaced (here via `rewriter.replaceOp`) |
| `failure()` | Pattern does not apply; op is left unchanged |

So “match” and “rewrite” are one step in this API: you decide in the same function whether the op fits and, if so, build new IR with `PatternRewriter`.

### How this pass relates to `one-shot-bufferize`

They solve **different** problems:

```text
┌─────────────────────┐     ┌──────────────────────────┐
│ tadd-lower-to-linalg │     │ one-shot-bufferize         │
│ (dialect conversion) │     │ (tensor → memref)          │
│ tadd → linalg/arith  │     │ memref.alloc, memref args  │
└─────────────────────┘     └──────────────────────────┘
         │                              │
         └────────── run in order ──────┘
```

On the command line, both passes are listed left to right: first rewrite inside each function, then bufferize the whole module.

**Driver detail:** `tadd-opt` must register **bufferization external models** for `tensor`, `linalg`, `arith`, and `func` (see `tadd/tools/tadd-opt.cpp`). Otherwise Stage 2 fails on `tensor.empty` even though Stage 1 succeeded. Full `mlir-opt` does that registration for you automatically.

### Mental model checklist

- `tadd.add` — domain-specific name for “tensor add”; good for parsing and teaching.
- **Lowering pass** — rewrite to `linalg.generic` **+** `arith.addf` so standard tools apply.
- **Bufferize** — turn **tensor values** into **memref buffers**; not part of the `tadd` dialect itself.
- **End goal of this experiment** — IR that uses the **memref** dialect at the function boundary, with the same element-wise add logic.

### Where to read the code

| File | What to look at |
|------|-----------------|
| `tadd/lib/LowerTAddToLinalg.cpp` | `LowerTAddAddOp::matchAndRewrite`, `LowerTAddToLinalgPass::runOnOperation` |
| `tadd/tools/tadd-opt.cpp` | `registerTAddLowerToLinalgPass()`, dialect + bufferization registration |
| `tadd/add.mlir` | Input |
| `tadd/add_memref.mlir` | Final memref form |

For general pass anatomy (`PassWrapper`, `OperationPass`, registration), see [§9](#9-experiment-5-writing-a-custom-pass). For the memref story in the rest of MLIR (not `tadd`-specific), see [§7](#7-experiment-3-lowering-to-memref).

---

## 12. Command Reference

| Task                      | Command                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| Parse and print MLIR      | `build/bin/mlir-opt file.mlir`                                                                          |
| Canonicalize              | `build/bin/mlir-opt file.mlir -canonicalize`                                                            |
| Bufferize tensor → memref | `build/bin/mlir-opt file.mlir -one-shot-bufferize="bufferize-function-boundaries"`                      |
| Lower to LLVM dialect     | `build/bin/mlir-opt file.mlir -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts` |
| Translate to LLVM IR      | `... | build/bin/mlir-translate -mlir-to-llvmir`                                                        |
| List available passes     | `build/bin/mlir-opt --help`                                                                             |
| List available dialects   | `build/bin/mlir-opt --show-dialects`                                                                    |
| Save output               | `build/bin/mlir-opt file.mlir -o out.mlir`                                                              |
| Run custom pass           | `build/bin/mlir-playground-opt file.mlir -list-dialects`                                                |
| Parse `tadd` dialect      | `build/bin/tadd-opt tadd/add.mlir`                                                                      |
| `tadd` → memref           | `build/bin/tadd-opt tadd/add.mlir -tadd-lower-to-linalg -one-shot-bufferize=bufferize-function-boundaries` |

---

## 13. Files Created

| File | Description |
|------|-------------|
| `simple.mlir` | Minimal scalar add: `func` + `arith` |
| `simple_memref.mlir` | Same logic via `memref.alloca` / `load` / `store` |
| `simple_tensor.mlir` | `linalg.fill` on `tensor`, bufferizes to memref |
| `mlirplayground/` | Custom pass skeleton (`list-dialects`) + driver |
| `tadd/` | Dialect `tadd` with one op: `tadd.add` on `tensor<1x8xf32>` |
| `tadd/add_memref.mlir` | `add.mlir` after `tadd-lower-to-linalg` + one-shot bufferize |
| `mlir_primer.md` | This document (includes integrated content from `MLIR_structure.md`) |

---

## Notes / Open Questions

- [ ] Lower `simple_memref.mlir` to LLVM IR (needs `-finalize-memref-to-llvm`)
- [ ] Run with `mlir-runner` to execute lowered code
- [ ] Explore `scf` loops and lowering to `cf`
- [ ] Write a custom pass (`mlirplayground` — `list-dialects`)
- [ ] Extend `list-dialects` to also report dialects used in types/attributes

---

*Last updated: 2026-10-25 (§9 restructured: pass basics, files/outputs, line-by-line, Q&A)*