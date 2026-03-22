// ---------------------------------------------------------------------------
// Available models — add new entries here as new models are trained.
// Each value must be a statically require()'d JSON file so Metro can bundle it.
// ---------------------------------------------------------------------------
export const MODELS = {
  'Meyda SVM v1': require('../assets/svm_model.json'),
};

export const MODEL_NAMES = Object.keys(MODELS);

// ---------------------------------------------------------------------------
// Precompute per-model caches once at load time
// ---------------------------------------------------------------------------
function buildCache(model) {
  const nClasses = model.classes.length;

  const svStart = new Array(nClasses).fill(0);
  for (let c = 1; c < nClasses; c++) svStart[c] = svStart[c - 1] + model.n_support[c - 1];

  const svNormSq = model.support_vectors.map(sv => {
    let s = 0; for (let i = 0; i < sv.length; i++) s += sv[i] * sv[i]; return s;
  });
  const svArrays = model.support_vectors.map(sv => new Float64Array(sv));
  const dualCoef = model.dual_coef.map(row => new Float64Array(row));

  return { nClasses, svStart, svNormSq, svArrays, dualCoef };
}

const CACHES = {};
for (const [name, data] of Object.entries(MODELS)) CACHES[name] = buildCache(data);

// ---------------------------------------------------------------------------
// Wu et al. (2004) pairwise coupling — direct port of libsvm's
// multiclass_probability(). Converts pairwise Platt probabilities into a
// proper k-class probability distribution.
//
// r[i][j] = P(class i | comparing i vs j), with r[i][i] = 0 and
// r[i][j] + r[j][i] = 1.
// ---------------------------------------------------------------------------
function multiclassProbability(k, r) {
  // Build Q matrix
  const Q  = Array.from({ length: k }, () => new Float64Array(k));
  for (let t = 0; t < k; t++) {
    Q[t][t] = 0;
    for (let j = 0; j < t; j++) {
      Q[t][t] += r[j][t] * r[j][t];
      Q[t][j]  = Q[j][t];
    }
    for (let j = t + 1; j < k; j++) {
      Q[t][t] += r[j][t] * r[j][t];
      Q[t][j]  = -r[t][j] * r[j][t];
    }
  }

  const p   = new Float64Array(k).fill(1 / k);
  const Qp  = new Float64Array(k);
  const eps = 0.005 / k;
  const maxIter = Math.max(100, k);

  for (let iter = 0; iter < maxIter; iter++) {
    let pQp = 0;
    for (let t = 0; t < k; t++) {
      Qp[t] = 0;
      for (let j = 0; j < k; j++) Qp[t] += Q[t][j] * p[j];
      pQp += p[t] * Qp[t];
    }

    let maxError = 0;
    for (let t = 0; t < k; t++) {
      const err = Math.abs(Qp[t] - pQp);
      if (err > maxError) maxError = err;
    }
    if (maxError < eps) break;

    for (let t = 0; t < k; t++) {
      const diff = (-Qp[t] + pQp) / Q[t][t];
      p[t] += diff;
      pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
      for (let j = 0; j < k; j++) {
        Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
        p[j] /= (1 + diff);
      }
      p[t] = (p[t] + 1) / (1 + diff);
    }
  }

  return p;
}

// ---------------------------------------------------------------------------
// SVM predict
//
// When the model includes Platt calibration params (prob_a / prob_b),
// confidence is a proper calibrated probability from Wu et al. coupling and
// probs is a { label: probability } map over all classes.
//
// When params are absent (old model), falls back to weighted decision-function
// magnitude and probs is undefined.
// ---------------------------------------------------------------------------
export function predict(features, modelName = MODEL_NAMES[0]) {
  const model  = MODELS[modelName];
  const { nClasses, svStart, svNormSq, svArrays, dualCoef } = CACHES[modelName];

  const n      = features.length;
  const scaled = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    scaled[i] = (features[i] - model.scaler_mean[i]) / model.scaler_scale[i];
  }

  let xNormSq = 0;
  for (let i = 0; i < n; i++) xNormSq += scaled[i] * scaled[i];

  const kernelVals = svArrays.map((sv, k) => {
    let dot = 0;
    for (let i = 0; i < n; i++) dot += scaled[i] * sv[i];
    return Math.exp(-model.gamma * (xNormSq - 2 * dot + svNormSq[k]));
  });

  const votes    = new Array(nClasses).fill(0);
  const dfWon    = new Array(nClasses).fill(0);
  const decVals  = new Float64Array(nClasses * (nClasses - 1) / 2);
  let pairIdx    = 0;

  for (let i = 0; i < nClasses; i++) {
    for (let j = i + 1; j < nClasses; j++) {
      let sum = model.intercept[pairIdx];
      for (let s = 0; s < model.n_support[i]; s++)
        sum += dualCoef[j - 1][svStart[i] + s] * kernelVals[svStart[i] + s];
      for (let s = 0; s < model.n_support[j]; s++)
        sum += dualCoef[i][svStart[j] + s] * kernelVals[svStart[j] + s];

      decVals[pairIdx] = sum;
      if (sum > 0) { votes[i]++; dfWon[i] += sum;  }
      else         { votes[j]++; dfWon[j] -= sum;  }
      pairIdx++;
    }
  }

  const winnerIdx = votes.indexOf(Math.max(...votes));

  // Platt calibration path
  if (model.prob_a && model.prob_b) {
    // Build pairwise probability matrix r[i][j] = P(class i | pair i,j)
    const r = Array.from({ length: nClasses }, () => new Float64Array(nClasses));
    let idx = 0;
    for (let i = 0; i < nClasses; i++) {
      for (let j = i + 1; j < nClasses; j++) {
        const raw = 1 / (1 + Math.exp(model.prob_a[idx] * decVals[idx] + model.prob_b[idx]));
        const rij = Math.min(Math.max(raw, 1e-7), 1 - 1e-7);
        r[i][j] = rij;
        r[j][i] = 1 - rij;
        idx++;
      }
    }

    const classProbs = multiclassProbability(nClasses, r);

    // Build { label: probability } map for probability-averaging majority vote
    const probs = {};
    for (let i = 0; i < nClasses; i++) probs[model.classes[i]] = classProbs[i];

    return {
      label:      model.classes[winnerIdx],
      confidence: classProbs[winnerIdx],
      probs,
    };
  }

  // Fallback: weighted decision-function confidence
  const totalDf = dfWon.reduce((a, b) => a + b, 0);
  return {
    label:      model.classes[winnerIdx],
    confidence: totalDf > 0 ? dfWon[winnerIdx] / totalDf : votes[winnerIdx] / (nClasses - 1),
  };
}

// ---------------------------------------------------------------------------
// Majority vote across all window predictions.
//
// When predictions carry Platt `probs`, averages the full probability
// distribution across windows (more robust than simple majority voting).
// Falls back to vote counting for models without calibration.
// ---------------------------------------------------------------------------
export function majorityVote(preds) {
  const filtered = preds.filter(p => p.label !== 'Background');
  if (!filtered.length) return { label: 'Background', confidence: 1, all: [] };

  // Vote-counting — more decisive than probability averaging for 52-class Platt models
  const counts = {};
  for (const { label } of filtered) counts[label] = (counts[label] || 0) + 1;
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const winner = sorted[0][0];
  const all    = sorted
    .map(([label, count]) => ({ label, confidence: count / filtered.length }))
    .slice(0, 5);
  return { label: winner, confidence: all[0].confidence, all };
}

// ---------------------------------------------------------------------------
// Per-species model accuracy (F1 on held-out test set)
// Returns null if the model was not trained with per_class_f1 export.
// ---------------------------------------------------------------------------
export function getModelF1(label, modelName = MODEL_NAMES[0]) {
  const f1 = MODELS[modelName].per_class_f1?.[label] ?? null;
  return f1 !== null && f1 > 0 ? f1 : null;
}

// ---------------------------------------------------------------------------
// Latin name lookup
// ---------------------------------------------------------------------------
export const LATIN_NAMES = {
  'Australian Lace-Lid':              'Nyctimystes dayi',
  'Baw Baw Frog':                     'Philoria frosti',
  'Beautiful Nursery Frog':           'Cophixalus concinnus',
  'Bellenden Ker Nursery Frog':       'Cophixalus neglectus',
  'Booroolong Frog':                  'Litoria booroolongensis',
  'Cave Frog':                        'Litoria cavernicola',
  'Common Eastern Froglet':           'Crinia signifera',
  'Davies Tree Frog':                 'Litoria daviesae',
  'Desert Spadefoot':                 'Notaden nichollsi',
  'Eastern Banjo Frog':               'Limnodynastes dumerilii',
  'Eungella Day Frog':                'Taudactylus eungellensis',
  'Flat Headed Frog':                 'Litoria inermis',
  "Fleay's Barred Frog":              'Mixophyes fleayi',
  'Giant Burrowing Frog':             'Heleioporus australiacus',
  'Green Tree Frog':                  'Litoria caerulea',
  'Green and Golden Bell Frog':       'Litoria aurea',
  "Hosmer's Nursery Frog":            'Cophixalus hosmeri',
  'Howard Springs Toadlet':           'Uperoleia inundata',
  'Kroombit Tops Tinker Frog':        'Taudactylus pleione',
  'Kuranda Tree Frog':                'Litoria myola',
  "Littlejohn's Toadlet":             'Uperoleia littlejohni',
  'Magnificent Brood Frog':           'Pseudophryne covacevichae',
  'Magnificent Tree Frog':            'Litoria splendida',
  "Mahony's Toadlet":                 'Uperoleia mahonyi',
  'Moss Froglet':                     'Bryobatrachus nimbus',
  'Motorbike Frog':                   'Litoria moorei',
  'Mount Top Nursery Frog':           'Cophixalus monticola',
  'Mountain Frog':                    'Philoria sphagnicolus',
  'Mountain Mist Frog':               'Litoria nyakalensis',
  'Mt Elliot Nursery Frog':           'Cophixalus mcdonaldi',
  'Northern Corroboree Frog':         'Pseudophryne pengilleyi',
  'Northern Flinders Ranges Froglet': 'Crinia flindersensis',
  'Northern Heath Frog':              'Litoria olongburensis',
  'Northern Snapping Frog':           'Cyclorana australis',
  'Northern Tinker Frog':             'Taudactylus rheophilus',
  'Orange-bellied Froglet':           'Crinia parinsignifera',
  'Pobblebonk':                       'Limnodynastes dumerilii',
  'Rattling Nursery Frog':            'Cophixalus rattus',
  'Southern Barred Frog':             'Mixophyes balbus',
  'Southern Bell Frog':               'Litoria raniformis',
  'Southern Corroboree Frog':         'Pseudophryne corroboree',
  'Southern Heath Frog':              'Litoria heinrichii',
  'Spotted Tree Frog':                'Litoria spenceri',
  'Sunset Frog':                      'Spicospina flammocaerulea',
  'Tapping Nursery Frog':             'Cophixalus bombiens',
  'Tasmanian Tree Frog':              'Litoria burrowsae',
  'Tusked Frog':                      'Adelotus brevis',
  'Victorian Smooth Froglet':         'Geocrinia victoriana',
  'Wallum Sedge Frog':                'Litoria olongburensis',
  'White Bellied Frog':               'Geocrinia alba',
  'Yellow Spotted Bell Frog':         'Litoria castanea',
};

export const getNames = (label) => ({ displayName: label, latin: LATIN_NAMES[label] ?? null });
