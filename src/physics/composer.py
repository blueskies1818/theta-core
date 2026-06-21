"""Per-domain template composer — composition by architecture, not training.

Replaces the single ExpressionSequenceModel with a pipeline:
  1. DomainClassifier: quantity set → domain scores (gravity, spring, EM)
  2. Per-domain DomainTemplateGenerator: domain-relevant quantities → template expression
  3. ExpressionComposer: algorithmic deduplication and concatenation of templates

When multiple domains activate, their templates UNION — the composer handles
deduplication of shared sub-expressions (e.g., ½mv² appears in both gravity and spring).

Architecture decision: composition is BY ARCHITECTURE. No cross-domain training.
Each domain model trains only on its own data.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Domain Definitions ─────────────────────────────────────────────────────────

DOMAINS = ["gravity", "spring", "em"]
COLLISION_DOMAIN = "collision"

# Which quantities are relevant per domain (quantities fed to template generator)
DOMAIN_QUANTITIES: dict[str, list[str]] = {
    "gravity":   ["m", "g", "h", "v"],
    "spring":    ["m", "k", "h", "v"],
    "em":        ["m", "g", "h", "v", "q", "E"],
    "collision": ["m", "v", "t"],
}

# Quantity key sets for domain detection (used by assign_domain_labels)
DOMAIN_QUANTITY_KEY: dict[str, set[str]] = {
    "gravity":   {"g"},
    "spring":    {"k"},
    "em":        {"q", "E"},
    "collision": set(),  # detected via scenario metadata, not quantity keys
}

# Hardcoded fallback templates (used when no generator is trained)
DOMAIN_TEMPLATES: dict[str, str] = {
    "gravity":   "m*g*h + 0.5*m*v^2",
    "spring":    "0.5*k*h^2 + 0.5*m*v^2",
    "em":        "q*E*h + 0.5*m*v^2",
    "collision": "0.5*m*v^2",
}

# Quantity vocab for the domain classifier feature vector
QUANTITY_VOCAB = [
    "m", "g", "h", "v", "t",
    "k", "L", "q", "E", "x", "y", "r",
]

QTY_TO_IDX = {q: i for i, q in enumerate(QUANTITY_VOCAB)}
NUM_QUANTITIES = len(QUANTITY_VOCAB)

# Shared token vocab for template generators (subset of the full vocab)
# We only need operators, constants, and quantities — no scenario types
TEMPLATE_SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
TEMPLATE_PAD_IDX = 0
TEMPLATE_SOS_IDX = 1
TEMPLATE_EOS_IDX = 2
TEMPLATE_UNK_IDX = 3

TEMPLATE_OPERATORS = ["+", "-", "*", "/", "^"]
TEMPLATE_CONSTANTS = ["0", "0.5", "1", "2", "1/2"]

TEMPLATE_TOKENS = (
    TEMPLATE_SPECIAL_TOKENS
    + TEMPLATE_OPERATORS
    + TEMPLATE_CONSTANTS
    + QUANTITY_VOCAB
)
TEMPLATE_TOKEN_TO_ID = {t: i for i, t in enumerate(TEMPLATE_TOKENS)}
TEMPLATE_ID_TO_TOKEN = {i: t for t, i in TEMPLATE_TOKEN_TO_ID.items()}
TEMPLATE_VOCAB_SIZE = len(TEMPLATE_TOKEN_TO_ID)


# ── Domain Classifier ──────────────────────────────────────────────────────────

class DomainClassifier(nn.Module):
    """Simple MLP: quantity set binary features → domain scores.

    Input: binary vector of length NUM_QUANTITIES (1 if quantity present)
    Output: 3 scores [gravity, spring, em] — sigmoid applied for thresholding.

    Parameters
    ----------
    hidden_dim : int
        Hidden layer dimension (default 32).
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(NUM_QUANTITIES, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)  # 3 domains
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, NUM_QUANTITIES] binary feature vector

        Returns:
            [batch, 3] raw logits for [gravity, spring, em]
        """
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.fc3(h)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for each domain.

        Returns [batch, 3] with values in [0, 1].
        """
        return torch.sigmoid(self.forward(x))

    def predict_domains(self, x: torch.Tensor, threshold: float = 0.5) -> list[list[str]]:
        """Predict active domains for each batch item.

        Args:
            x: [batch, NUM_QUANTITIES]
            threshold: probability threshold for domain activation

        Returns:
            List of lists of active domain names.
        """
        probs = self.predict_proba(x)  # [batch, 3]
        active = probs > threshold
        results: list[list[str]] = []
        for b in range(active.size(0)):
            domains = [DOMAINS[i] for i in range(3) if active[b, i].item()]
            results.append(domains)
        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Domain Template Generator ──────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for template generator."""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DomainTemplateGenerator(nn.Module):
    """Small encoder-decoder transformer for a single physics domain.

    Maps domain-relevant quantity symbols → template expression tokens.
    ~50K parameters. Trained only on its domain's data.

    Parameters
    ----------
    d_model : int
        Hidden dimension (default 40, for ~48K total params).
    nhead : int
        Attention heads (default 2).
    num_encoder_layers : int
        Encoder layers (default 1).
    num_decoder_layers : int
        Decoder layers (default 1).
    max_src_len : int
        Max source (quantity list) length.
    max_tgt_len : int
        Max target (expression) length.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 40,
        nhead: int = 2,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        max_src_len: int = 8,
        max_tgt_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = TEMPLATE_VOCAB_SIZE

        self.token_embedding = nn.Embedding(
            TEMPLATE_VOCAB_SIZE, d_model, padding_idx=TEMPLATE_PAD_IDX
        )

        self.src_pos_encoding = PositionalEncoding(d_model, max_src_len, dropout)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_tgt_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.output_proj = nn.Linear(d_model, TEMPLATE_VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,          # [batch, src_len]
        tgt: torch.Tensor,          # [batch, tgt_len]
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass. Returns logits: [batch, tgt_len, vocab_size]."""
        src_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.src_pos_encoding(src_emb)

        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)

        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        return self.output_proj(output)

    def encode_source(
        self, src: torch.Tensor, src_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        src_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.src_pos_encoding(src_emb)
        return self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(output)

    def generate(
        self,
        src: torch.Tensor,
        src_padding_mask: torch.Tensor | None = None,
        max_len: int = 32,
        temperature: float = 0.0,
    ) -> list[list[int]]:
        """Generate template expression for a batch of source inputs.

        Greedy decoding when temperature=0.
        """
        batch_size = src.size(0)
        device = src.device

        memory = self.encode_source(src, src_padding_mask)
        generated = torch.full(
            (batch_size, 1), TEMPLATE_SOS_IDX, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_len = generated.size(1)
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device) * float("-inf"), diagonal=1
            )
            logits = self.decode_step(
                generated, memory, tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if temperature > 0:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = next_logits.argmax(dim=-1)

            next_token = torch.where(finished, TEMPLATE_PAD_IDX, next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == TEMPLATE_EOS_IDX)
            if finished.all():
                break

        results: list[list[int]] = []
        for b in range(batch_size):
            seq = generated[b].tolist()
            if TEMPLATE_EOS_IDX in seq:
                seq = seq[:seq.index(TEMPLATE_EOS_IDX) + 1]
            results.append(seq)
        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CollisionTemplateGenerator(DomainTemplateGenerator):
    """Template generator specifically for collision physics domain.

    Subclasses DomainTemplateGenerator with collision-specific defaults.
    Trained on 7 collision scenarios (elastic, inelastic, 1D, 2D).
    Learns templates: ½mv² (kinetic energy, piecewise-constant across impact),
    m₁v₁+m₂v₂ (momentum, conserved in elastic).

    Collision-specific behavior:
      - Kinetic energy is piecewise-constant (constant before and after
        impact, with a step change at the collision instant)
      - Momentum is conserved in elastic collisions
      - Energy is partially conserved in inelastic collisions

    Parameters
    ----------
    d_model : int
        Hidden dimension (default 36, for ~50K total params).
    nhead : int
        Attention heads (default 2).
    num_encoder_layers : int
        Encoder layers (default 1).
    num_decoder_layers : int
        Decoder layers (default 1).
    max_src_len : int
        Max source length.
    max_tgt_len : int
        Max target length.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 36,
        nhead: int = 2,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        max_src_len: int = 8,
        max_tgt_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            dropout=dropout,
        )


# ── Template Tokenization ──────────────────────────────────────────────────────

def tokenize_expression(expr_str: str) -> list[int]:
    """Tokenize expression string into template token IDs."""
    tokens: list[int] = []
    i = 0
    s = expr_str.strip()
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        matched = False
        for length in (4, 3, 2, 1):
            if i + length <= len(s):
                candidate = s[i:i + length]
                if candidate in TEMPLATE_TOKEN_TO_ID:
                    tokens.append(TEMPLATE_TOKEN_TO_ID[candidate])
                    i += length
                    matched = True
                    break
        if matched:
            continue
        ch = s[i]
        if ch in TEMPLATE_TOKEN_TO_ID:
            tokens.append(TEMPLATE_TOKEN_TO_ID[ch])
        else:
            tokens.append(TEMPLATE_UNK_IDX)
        i += 1
    return tokens


def detokenize_expression(token_ids: list[int]) -> str:
    """Convert template token IDs back to expression string."""
    parts: list[str] = []
    for tid in token_ids:
        if tid in (TEMPLATE_PAD_IDX, TEMPLATE_SOS_IDX):
            continue
        if tid == TEMPLATE_EOS_IDX:
            break
        tok = TEMPLATE_ID_TO_TOKEN.get(tid, "<unk>")
        if tok in TEMPLATE_SPECIAL_TOKENS:
            continue
        parts.append(tok)
    return "".join(parts) if parts else ""


def expression_to_tensor(expr_str: str, max_len: int = 32) -> torch.Tensor:
    """Convert expression string to padded tensor [SOS, tokens..., EOS, PAD...]."""
    tokens = tokenize_expression(expr_str)
    ids = [TEMPLATE_SOS_IDX] + tokens + [TEMPLATE_EOS_IDX]
    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [TEMPLATE_EOS_IDX]
    padded = ids + [TEMPLATE_PAD_IDX] * (max_len - len(ids))
    return torch.tensor(padded, dtype=torch.long)


def quantities_to_tensor(
    quantity_symbols: list[str], max_len: int = 8
) -> torch.Tensor:
    """Convert quantity symbols to padded tensor of token IDs."""
    ids = [TEMPLATE_TOKEN_TO_ID.get(q, TEMPLATE_UNK_IDX) for q in quantity_symbols]
    if len(ids) > max_len:
        ids = ids[:max_len]
    padded = ids + [TEMPLATE_PAD_IDX] * (max_len - len(ids))
    return torch.tensor(padded, dtype=torch.long)


def quantities_to_features(quantity_symbols: list[str]) -> torch.Tensor:
    """Convert quantity symbols to binary feature vector [NUM_QUANTITIES]."""
    vec = torch.zeros(NUM_QUANTITIES)
    for q in quantity_symbols:
        idx = QTY_TO_IDX.get(q)
        if idx is not None:
            vec[idx] = 1.0
    return vec


# ── Expression Composer ────────────────────────────────────────────────────────

class ExpressionComposer:
    """Algorithmic composer: takes domain templates, deduplicates, concatenates.

    Composition is BY ARCHITECTURE — not learned. When multiple domains
    activate, their template terms are UNION-ed with deduplication of
    shared sub-expressions (e.g., 0.5*m*v^2 appears in multiple domains).
    """

    @staticmethod
    def compose(templates: list[str]) -> str:
        """Compose multiple template expressions into one.

        Args:
            templates: List of expression strings from active domain generators.

        Returns:
            Combined expression string with deduplicated terms.
        """
        if not templates:
            return ""
        if len(templates) == 1:
            return templates[0]

        # Parse each template into terms
        all_terms: list[str] = []
        for tmpl in templates:
            terms = ExpressionComposer._parse_terms(tmpl)
            all_terms.extend(terms)

        # Deduplicate by canonical form
        unique_terms = ExpressionComposer._deduplicate(all_terms)

        # Combine with '+'
        combined = ExpressionComposer._join_terms(unique_terms)
        return ExpressionComposer._clean(combined)

    @staticmethod
    def _parse_terms(expr: str) -> list[str]:
        """Split expression on top-level '+' and '-' into terms."""
        terms: list[str] = []
        depth = 0
        current: list[str] = []
        i = 0
        # Clean up spaces
        expr = expr.strip()
        while i < len(expr):
            ch = expr[i]
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == '+' and depth == 0:
                terms.append("".join(current).strip())
                current = []
            elif ch == '-' and depth == 0 and i > 0 and expr[i - 1] not in '+-*/^(':
                terms.append("".join(current).strip())
                current = ["-"]
            else:
                current.append(ch)
            i += 1
        if current:
            terms.append("".join(current).strip())
        return [t for t in terms if t]

    @staticmethod
    def _canonicalize(term: str) -> str:
        """Canonicalize a term for deduplication.

        Sorts multiplicative factors alphabetically, handles leading sign.
        """
        term = term.strip()
        sign = ""
        if term.startswith("-"):
            sign = "-"
            term = term[1:].strip()

        # Split on '*' to get factors
        factors = [f.strip() for f in term.split("*")]
        # Sort factors (this handles commutativity)
        factors.sort()

        # Combine sign with coefficient
        if factors:
            # Move numeric/constant factors to front
            numeric = [f for f in factors if _is_numeric_or_constant(f)]
            variables = [f for f in factors if not _is_numeric_or_constant(f)]
            sorted_factors = numeric + variables
            return sign + "*".join(sorted_factors)
        return sign + term

    @staticmethod
    def _deduplicate(terms: list[str]) -> list[str]:
        """Deduplicate terms by canonical form. Preserves order of first appearance."""
        seen: set[str] = set()
        unique: list[str] = []
        for term in terms:
            canonical = ExpressionComposer._canonicalize(term)
            if canonical not in seen:
                seen.add(canonical)
                unique.append(term)
        return unique

    @staticmethod
    def _join_terms(terms: list[str]) -> str:
        """Join terms with '+', handling leading signs."""
        if not terms:
            return ""
        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += " - " + term[1:]
            else:
                result += " + " + term
        return result

    @staticmethod
    def _clean(expr: str) -> str:
        """Clean up spacing and formatting."""
        # Remove redundant spaces
        expr = re.sub(r'\s+', '', expr)
        # Put spaces around + and -
        expr = expr.replace("+", " + ").replace("-", " - ")
        # Fix double-space
        expr = re.sub(r'\s+', ' ', expr).strip()
        return expr


def _is_numeric_or_constant(s: str) -> bool:
    """Check if string is a numeric value or known constant."""
    try:
        float(s)
        return True
    except ValueError:
        pass
    # Check fractions like "1/2"
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                float(parts[0])
                float(parts[1])
                return True
            except ValueError:
                pass
    return s in {"g", "0.5", "2", "1/2"}


# ── Full Pipeline ──────────────────────────────────────────────────────────────

class PerDomainComposer(nn.Module):
    """Full per-domain composition pipeline.

    Combines DomainClassifier, per-domain template generators, and
    algorithmic ExpressionComposer into a single callable interface.

    Parameters
    ----------
    domain_classifier : DomainClassifier
    template_generators : dict[str, DomainTemplateGenerator]
        Mapping from domain name to its template generator.
    composer : ExpressionComposer
        Algorithmic term composer.
    threshold : float
        Domain activation threshold (default 0.5).
    """

    def __init__(
        self,
        domain_classifier: DomainClassifier,
        template_generators: dict[str, DomainTemplateGenerator],
        composer: ExpressionComposer | None = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.domain_classifier = domain_classifier
        self.template_generators = nn.ModuleDict(template_generators)
        self.composer = composer or ExpressionComposer()
        self.threshold = threshold
        self._device = torch.device("cpu")

    def to(self, *args, **kwargs):  # type: ignore[override]
        result = super().to(*args, **kwargs)
        if args and isinstance(args[0], torch.device):
            self._device = args[0]
        elif "device" in kwargs:
            self._device = kwargs["device"]
        return result

    def forward(
        self,
        quantity_symbols: list[str],
        temperature: float = 0.0,
        scenario_id: str | None = None,
        phase_regions: list[dict] | None = None,
    ) -> tuple[str, list[str]]:
        """Full pipeline: classify domain → generate templates → compose.

        Args:
            quantity_symbols: List of quantity variable names present.
            temperature: Generation temperature (0 = greedy).
            scenario_id: Optional scenario ID for collision detection.
            phase_regions: Optional phase regions for collision detection.

        Returns:
            (composed_expression, active_domains) tuple.
        """
        # Step 1: Classify domain
        features = quantities_to_features(quantity_symbols).unsqueeze(0).to(self._device)
        active_domains = self.domain_classifier.predict_domains(features, self.threshold)[0]

        # Fallback: if no domain active, use heuristic based on quantities
        if not active_domains:
            active_domains = self._heuristic_domains(quantity_symbols)

        # Secondary pass: with lower threshold for rare domains (EM)
        all_probs = self.domain_classifier.predict_proba(features).squeeze(0)
        low_threshold = max(0.15, self.threshold * 0.4)
        for i, domain in enumerate(DOMAINS):
            if domain not in active_domains and all_probs[i].item() > low_threshold:
                active_domains.append(domain)

        # Collision detection: activate collision domain when phase_regions
        # or collision-named scenario is present
        is_collision = self._detect_collision(scenario_id, phase_regions)
        if is_collision and COLLISION_DOMAIN not in active_domains:
            active_domains.append(COLLISION_DOMAIN)

        # Step 2: Generate templates for each active domain
        templates: list[str] = []
        for domain in active_domains:
            if domain in self.template_generators:
                gen = self.template_generators[domain]
                domain_qties = self._filter_domain_quantities(
                    quantity_symbols, domain
                )
                if not domain_qties:
                    continue
                src = quantities_to_tensor(domain_qties, max_len=8).unsqueeze(0).to(self._device)
                src_mask = (src == TEMPLATE_PAD_IDX)
                with torch.no_grad():
                    gen_ids = gen.generate(
                        src, src_padding_mask=src_mask,
                        max_len=32, temperature=temperature,
                    )
                tmpl = detokenize_expression(gen_ids[0])
                if tmpl:
                    templates.append(tmpl)

        if not templates:
            return "", active_domains

        # Step 3: Compose
        composed = self.composer.compose(templates)

        # Post-filter: remove terms that reference quantities not in the input
        composed = self._filter_valid_terms(composed, quantity_symbols)

        return composed, active_domains

    @staticmethod
    def _filter_valid_terms(expr: str, available_quantities: list[str]) -> str:
        """Remove terms that reference quantities not in the available set."""
        if not expr:
            return ""
        terms = ExpressionComposer._parse_terms(expr)
        valid_terms: list[str] = []
        avail = set(available_quantities)
        for term in terms:
            # Extract symbols from the term
            import re
            term_symbols = set(re.findall(r'[a-zA-Z]', term))
            # Only keep terms whose symbols are all available
            if term_symbols.issubset(avail):
                valid_terms.append(term)
        if not valid_terms:
            return expr  # keep original if all filtered out
        return ExpressionComposer._join_terms(valid_terms)

    @staticmethod
    def _filter_domain_quantities(
        all_quantities: list[str], domain: str
    ) -> list[str]:
        """Filter quantities to only those relevant for a domain.

        Only includes quantities that are actually present in the input.
        Skips domains whose core quantities are completely absent.
        """
        relevant = set(DOMAIN_QUANTITIES.get(domain, []))
        filtered = [q for q in all_quantities if q in relevant]
        return filtered  # Empty list = skip this domain

    @staticmethod
    def _heuristic_domains(quantity_symbols: list[str]) -> list[str]:
        """Fallback domain detection from quantity set."""
        syms = set(quantity_symbols)
        domains: list[str] = []
        if "g" in syms and "m" in syms:
            domains.append("gravity")
        if "k" in syms:
            domains.append("spring")
        if "q" in syms or "E" in syms:
            domains.append("em")
        return domains if domains else ["gravity"]  # default

    @staticmethod
    def _detect_collision(
        scenario_id: str | None,
        phase_regions: list[dict] | None,
    ) -> bool:
        """Detect whether a scenario involves collision physics.

        Uses scenario ID naming convention and phase_regions metadata.
        Collision scenarios have phase_regions with 'before_collision'
        or 'after_collision' labels.
        """
        if scenario_id and "collision" in scenario_id.lower():
            return True
        if phase_regions:
            for region in phase_regions:
                label = region.get("label", "")
                if "collision" in label.lower():
                    return True
        return False

    def count_parameters(self) -> int:
        total = self.domain_classifier.count_parameters()
        for gen in self.template_generators.values():
            total += gen.count_parameters()
        return total


# ── Save/Load ──────────────────────────────────────────────────────────────────

def save_composer(composer: PerDomainComposer, checkpoint_dir: str | Path) -> None:
    """Save all composer components to disk."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save domain classifier
    torch.save(
        {"model_state_dict": composer.domain_classifier.state_dict()},
        checkpoint_dir / "domain_classifier.pt",
    )

    # Save template generators
    for domain, gen in composer.template_generators.items():
        torch.save(
            {"model_state_dict": gen.state_dict()},
            checkpoint_dir / f"{domain}_template.pt",
        )


def load_composer(
    checkpoint_dir: str | Path,
    device: torch.device = torch.device("cpu"),
) -> PerDomainComposer:
    """Load all composer components from disk."""
    checkpoint_dir = Path(checkpoint_dir)

    # Load domain classifier
    clf = DomainClassifier()
    clf_ckpt = torch.load(
        checkpoint_dir / "domain_classifier.pt",
        map_location=device, weights_only=False,
    )
    clf.load_state_dict(clf_ckpt["model_state_dict"])
    clf = clf.to(device)
    clf.eval()

    # Load template generators
    generators: dict[str, DomainTemplateGenerator] = {}
    for domain in DOMAINS:
        gen_path = checkpoint_dir / f"{domain}_template.pt"
        if gen_path.exists():
            gen_ckpt = torch.load(
                gen_path, map_location=device, weights_only=False,
            )
            d_model = gen_ckpt.get("d_model", 40)
            nhead = gen_ckpt.get("nhead", 2)
            gen = DomainTemplateGenerator(d_model=d_model, nhead=nhead)
            gen.load_state_dict(gen_ckpt["model_state_dict"])
        else:
            gen = DomainTemplateGenerator()
        gen = gen.to(device)
        gen.eval()
        generators[domain] = gen

    # Load collision template generator if available
    collision_path = checkpoint_dir / "collision_template.pt"
    if collision_path.exists():
        col_ckpt = torch.load(
            collision_path, map_location=device, weights_only=False,
        )
        d_model = col_ckpt.get("d_model", 36)
        nhead = col_ckpt.get("nhead", 2)
        col_gen = CollisionTemplateGenerator(d_model=d_model, nhead=nhead)
        col_gen.load_state_dict(col_ckpt["model_state_dict"])
        col_gen = col_gen.to(device)
        col_gen.eval()
        generators[COLLISION_DOMAIN] = col_gen

    composer = PerDomainComposer(clf, generators)
    composer._device = device
    return composer


# ── Backward-compatible aliases and thin wrappers ─────────────────────────────

# Alias for tests/training scripts
quantity_set_to_features = quantities_to_features


def prepare_source_tensor(
    quantity_symbols: list[str],
    max_src_len: int = 16,
) -> torch.Tensor:
    """Convert quantity symbols to a source tensor for template generators.

    Returns tensor of shape [1, max_src_len] compatible with DomainTemplateGenerator.
    """
    return quantities_to_tensor(quantity_symbols, max_len=max_src_len).unsqueeze(0)


def assign_domain_labels(quantity_symbols: list[str]) -> list[int]:
    """Assign multi-label domain indicators based on quantity symbols.

    Returns list of 0/1 for [gravity, spring, em].
    A scenario like mass_spring_gravity has g AND k → [1, 1, 0].
    """
    syms = set(quantity_symbols)
    return [
        1 if DOMAIN_QUANTITY_KEY["gravity"] & syms or "g" in syms else 0,
        1 if DOMAIN_QUANTITY_KEY["spring"] & syms else 0,
        1 if DOMAIN_QUANTITY_KEY["em"] & syms else 0,
    ]


def extract_domain_examples(
    observations_path: str | Path,
    domain: str,
) -> list[dict]:
    """Extract training examples for a single domain from the observation database.

    Each example: {quantities: dict, expression: str}

    Only includes observations whose known_invariant matches the domain.
    For EM domain, checks both quantities and parameters for q/E symbols.
    For collision domain, detects via collision-named scenario IDs and
    phase_regions with collision labels.
    """
    import json

    with open(observations_path) as f:
        data = json.load(f)

    examples: list[dict] = []
    for obs in data:
        inv = obs.get("known_invariant")
        qty_symbols = list(obs["quantities"].keys())
        # For domain detection, also consider parameter keys
        all_keys = set(qty_symbols) | set(obs.get("parameters", {}).keys())
        labels = _assign_domain_labels_from_keys(all_keys, obs)

        if domain == COLLISION_DOMAIN:
            # Collision detection: scenario ID or phase_regions
            oid = obs.get("id", "").lower()
            phase_regions = obs.get("phase_regions")
            is_collision = "collision" in oid
            if not is_collision and phase_regions:
                for region in phase_regions:
                    if "collision" in region.get("label", "").lower():
                        is_collision = True
                        break
            if is_collision and inv:
                examples.append({
                    "quantities": obs["quantities"],
                    "expression": inv,
                })
            continue

        domain_idx = DOMAINS.index(domain)
        if labels[domain_idx] == 1:
            # For EM domain, merge q/E from parameters into quantities
            quantities = dict(obs["quantities"])
            if domain == "em":
                if "q" not in all_keys or "E" not in all_keys:
                    continue
                # Add q/E from parameters if they exist there
                params = obs.get("parameters", {})
                for k in ("q", "E"):
                    if k in params and k not in quantities:
                        quantities[k] = "Scalar"
            examples.append({
                "quantities": quantities,
                "expression": inv,
            })

    return examples


def _assign_domain_labels_from_keys(
    all_keys: set[str], obs: dict | None = None
) -> list[int]:
    """Like assign_domain_labels but takes a set of all available keys.

    Optionally takes an observation dict for collision detection via
    ID and phase_regions metadata.
    """
    labels = [
        1 if DOMAIN_QUANTITY_KEY["gravity"] & all_keys or "g" in all_keys else 0,
        1 if DOMAIN_QUANTITY_KEY["spring"] & all_keys else 0,
        1 if DOMAIN_QUANTITY_KEY["em"] & all_keys else 0,
    ]
    # Collision is NOT part of the 3-class label — it's handled separately
    return labels


def save_domain_classifier(
    classifier: DomainClassifier, path: str | Path
) -> None:
    """Save domain classifier checkpoint."""
    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
        },
        path,
    )


def load_domain_classifier(path: str | Path) -> DomainClassifier:
    """Load domain classifier checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DomainClassifier()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def save_domain_generator(
    generator: DomainTemplateGenerator, path: str | Path
) -> None:
    """Save domain template generator checkpoint."""
    torch.save(
        {
            "model_state_dict": generator.state_dict(),
            "d_model": generator.d_model,
            "nhead": generator.encoder.layers[0].self_attn.num_heads,
        },
        path,
    )


def load_domain_generator(path: str | Path) -> DomainTemplateGenerator:
    """Load domain template generator checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    d_model = ckpt.get("d_model", 40)
    nhead = ckpt.get("nhead", 2)
    model = DomainTemplateGenerator(d_model=d_model, nhead=nhead)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# Module-level term manipulation wrappers (expose static methods)
_split_sum_terms = ExpressionComposer._parse_terms
_canonicalize_term = ExpressionComposer._canonicalize


def _terms_deduplicate(expressions: list[str]) -> list[str]:
    """Deduplicate terms across multiple expressions.

    Splits each expression into terms, canonicalizes, and returns union.
    """
    all_terms: list[str] = []
    for expr in expressions:
        if not expr or not expr.strip():
            continue
        terms = _split_sum_terms(expr)
        all_terms.extend(terms)
    return ExpressionComposer._deduplicate(all_terms)
