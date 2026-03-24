import "./global.css";
import { useState, useEffect, useRef, useMemo } from 'react';
import {
  View, Text, TouchableOpacity, StatusBar,
  StyleSheet, ScrollView, Dimensions, Animated,
  LayoutAnimation, Platform, UIManager,
} from 'react-native';

if (Platform.OS === 'android') {
  UIManager.setLayoutAnimationEnabledExperimental?.(true);
}
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import {
  useFonts,
  Inter_400Regular, Inter_500Medium, Inter_600SemiBold, Inter_700Bold,
} from '@expo-google-fonts/inter';
import { PlayfairDisplay_700Bold } from '@expo-google-fonts/playfair-display';

import { C, T } from './src/theme';
import { TEAM, SIGHTINGS } from './src/constants';
import { getNames, getModelF1, MODEL_NAMES } from './src/classifier';
import { useAudioProcessor } from './src/useAudioProcessor';

const { width: SW } = Dimensions.get('window');
const CARD_INNER_W = SW - 40 - 32;
const BAR_COUNT    = 36;


// ---------------------------------------------------------------------------
// StatusTicker — fades between status messages
// ---------------------------------------------------------------------------
function StatusTicker({ text }) {
  const opacity = useRef(new Animated.Value(1)).current;
  const [shown, setShown] = useState(text);

  useEffect(() => {
    Animated.timing(opacity, { toValue: 0, duration: 80, useNativeDriver: true }).start(() => {
      setShown(text);
      Animated.timing(opacity, { toValue: 1, duration: 120, useNativeDriver: true }).start();
    });
  }, [text]);

  return (
    <Animated.Text style={{ opacity, color: C.green, fontFamily: 'Inter-Medium', fontSize: 12, marginTop: 5 }}>
      {shown}
    </Animated.Text>
  );
}

// ---------------------------------------------------------------------------
// SelectionWaveform — drag to select region, tap to clear
// ---------------------------------------------------------------------------
const FROG_UNPLAYED  = 'rgba(58,104,48,0.50)';
const DRAG_THRESHOLD = 6;

function SelectionWaveform({
  samples, frogBuckets, playbackFraction,
  selection, onSelectionChange, onGestureActiveChange,
}) {
  const [w, setW] = useState(0);
  const pressStartX    = useRef(null);
  const pressStartFrac = useRef(null);
  const isDragging     = useRef(false);

  const fracAt = (x) => Math.max(0, Math.min(1, x / w));

  const handleGrant = (e) => {
    pressStartX.current    = e.nativeEvent.locationX;
    pressStartFrac.current = fracAt(e.nativeEvent.locationX);
    isDragging.current     = false;
    onGestureActiveChange?.(true);
  };

  const handleMove = (e) => {
    if (!w) return;
    const dx = Math.abs(e.nativeEvent.locationX - pressStartX.current);
    if (dx > DRAG_THRESHOLD || isDragging.current) {
      isDragging.current = true;
      const cur   = fracAt(e.nativeEvent.locationX);
      const start = Math.min(pressStartFrac.current, cur);
      const end   = Math.max(pressStartFrac.current, cur);
      onSelectionChange?.({ start, end });
    }
  };

  const handleRelease = () => {
    onGestureActiveChange?.(false);
    if (!isDragging.current) {
      // tap with no drag — clear selection
      onSelectionChange?.(null);
    }
    pressStartX.current = null;
  };

  const hasSel = !!selection && w > 0;

  return (
    <View
      onLayout={e => setW(e.nativeEvent.layout.width)}
      onStartShouldSetResponder={() => true}
      onMoveShouldSetResponder={() => true}
      onResponderGrant={handleGrant}
      onResponderMove={handleMove}
      onResponderRelease={handleRelease}
      onResponderTerminate={() => onGestureActiveChange?.(false)}
      style={{ height: 44, justifyContent: 'flex-end' }}
    >
      <View style={{ flexDirection: 'row', alignItems: 'flex-end', height: 36, gap: 1.5 }}>
        {samples.map((amp, i) => {
          const frac   = (i + 0.5) / samples.length;
          const played = frac < playbackFraction;
          const frog   = frogBuckets?.[i];
          const inSel  = hasSel && frac >= selection.start && frac <= selection.end;
          const bg     = frog
            ? (played ? C.green : FROG_UNPLAYED)
            : (played ? C.textLight : C.border);
          return (
            <View key={i} style={{
              flex: 1, height: Math.max(2, amp * 36), borderRadius: 1,
              backgroundColor: bg,
              opacity: hasSel ? (inSel ? 1 : 0.38) : 1,
            }} />
          );
        })}
      </View>

      {/* Selection region overlay */}
      {hasSel && (
        <View pointerEvents="none" style={{
          position: 'absolute', top: 0, bottom: 0,
          left: selection.start * w,
          width: Math.max(2, (selection.end - selection.start) * w),
          backgroundColor: 'rgba(58,104,48,0.10)',
          borderWidth: 1, borderColor: 'rgba(58,104,48,0.40)',
          borderRadius: 2,
        }} />
      )}

      {/* Playback cursor */}
      {playbackFraction > 0 && w > 0 && (
        <View style={{
          position: 'absolute', top: 0, bottom: 0,
          left: playbackFraction * w - 1, width: 2,
          borderRadius: 1, backgroundColor: C.text,
        }} />
      )}
    </View>
  );
}

// ---------------------------------------------------------------------------
// ScrubBar — horizontal slider for seeking
// ---------------------------------------------------------------------------
function ScrubBar({ value, totalSecs, onSeek, onGestureActiveChange }) {
  const [w, setW] = useState(0);
  const dragging = useRef(false);

  const fracAt = (x) => Math.max(0, Math.min(1, x / w));
  const seek   = (x) => { if (w) onSeek?.(fracAt(x)); };

  const fmt = (s) => {
    const t = Math.round(s);
    return `${Math.floor(t / 60)}:${(t % 60).toString().padStart(2, '0')}`;
  };

  return (
    <View style={{ marginTop: 10 }}>
      <View
        onLayout={e => setW(e.nativeEvent.layout.width)}
        onStartShouldSetResponder={() => true}
        onMoveShouldSetResponder={() => true}
        onResponderGrant={e => { dragging.current = true; onGestureActiveChange?.(true); seek(e.nativeEvent.locationX); }}
        onResponderMove={e => { if (dragging.current) seek(e.nativeEvent.locationX); }}
        onResponderRelease={() => { dragging.current = false; onGestureActiveChange?.(false); }}
        onResponderTerminate={() => { dragging.current = false; onGestureActiveChange?.(false); }}
        style={{ height: 28, justifyContent: 'center' }}
      >
        {/* Track */}
        <View style={{ height: 3, backgroundColor: C.border, borderRadius: 99, overflow: 'hidden' }}>
          <View style={{ height: '100%', width: `${value * 100}%`, backgroundColor: C.green, borderRadius: 99 }} />
        </View>
        {/* Thumb */}
        {w > 0 && (
          <View style={{
            position: 'absolute',
            left: Math.max(0, Math.min(w - 14, value * w - 7)),
            top: 7,
            width: 14, height: 14, borderRadius: 7,
            backgroundColor: C.text,
            shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
            shadowOpacity: 0.18, shadowRadius: 2, elevation: 3,
          }} />
        )}
      </View>
      {/* Time labels */}
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginTop: 2 }}>
        <Text style={[T.label, { color: C.textLight, fontSize: 9, fontVariant: ['tabular-nums'] }]}>
          {fmt(value * totalSecs)}
        </Text>
        <Text style={[T.label, { color: C.textLight, fontSize: 9, fontVariant: ['tabular-nums'] }]}>
          {fmt(totalSecs)}
        </Text>
      </View>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Waveform
// ---------------------------------------------------------------------------
function Waveform({ isActive, audioLevel = 0 }) {
  const bars  = useRef(Array.from({ length: BAR_COUNT }, () => new Animated.Value(0.05))).current;
  // Per-bar gain: arch shape + small fixed random offset so bars differ
  const gains = useRef(bars.map((_, i) => {
    const t = (i - BAR_COUNT / 2) / (BAR_COUNT / 2);
    return 0.45 + 0.55 * (1 - t * t) + (Math.random() * 0.15 - 0.075);
  })).current;

  useEffect(() => {
    if (!isActive) {
      bars.forEach(b => Animated.spring(b, { toValue: 0.05, useNativeDriver: true, speed: 4, bounciness: 0 }).start());
      return;
    }
    bars.forEach((bar, i) => {
      const target = Math.max(0.05, Math.min(1, audioLevel * gains[i]));
      Animated.spring(bar, { toValue: target, speed: 50, bounciness: 1, useNativeDriver: true }).start();
    });
  }, [audioLevel, isActive]);

  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', height: 56 }}>
      {bars.map((bar, i) => (
        <Animated.View key={i} style={{
          width: 3, height: 56, borderRadius: 2,
          backgroundColor: C.pill,
          transform: [{ scaleY: bar }],
        }} />
      ))}
    </View>
  );
}

// ---------------------------------------------------------------------------
// AppContent
// ---------------------------------------------------------------------------
const TABS = [
  { label: 'About',     page: 0 },
  { label: 'Record',    page: 1 },
  { label: 'Sightings', page: 2 },
];

function AppContent() {
  const [fontsLoaded] = useFonts({
    'Inter-Regular':        Inter_400Regular,
    'Inter-Medium':         Inter_500Medium,
    'Inter-SemiBold':       Inter_600SemiBold,
    'Inter-Bold':           Inter_700Bold,
    'PlayfairDisplay-Bold': PlayfairDisplay_700Bold,
  });

  const [activePage, setActivePage] = useState(1);  // start on Record
  const selectedModel = MODEL_NAMES[0];
  const [recSeconds,          setRecSeconds]          = useState(0);
  const [pendingRecord,       setPendingRecord]       = useState(false);
  const [waveSelection,       setWaveSelection]       = useState(null);
  const [waveGestureActive,   setWaveGestureActive]   = useState(false);
  const scrollRef   = useRef(null);
  const recTimerRef = useRef(null);

  const {
    prediction, status, progress, audioDuration, isProcessing,
    expandedResults, setExpandedResults, isRecording,
    audioLevel, waveformSamples, waveformFrogBuckets, waveformFrogLabels,
    playbackPositionSecs, seekAudio, audioUri, isPlaying,
    progressAnim, progressFadeAnim, resultDropAnim,
    startRecording, stopRecording, uploadAudio, playAudio, cancel,
  } = useAudioProcessor(selectedModel);

  const goToPage = (page) => {
    setActivePage(page);
    scrollRef.current?.scrollTo({ x: page * SW, animated: true });
  };

  // Start on Record tab (page 1)
  useEffect(() => {
    setTimeout(() => scrollRef.current?.scrollTo({ x: SW, animated: false }), 0);
  }, []);

  useEffect(() => {
    if (isRecording) {
      setPendingRecord(false);
      setRecSeconds(0);
      recTimerRef.current = setInterval(() => setRecSeconds(s => s + 1), 1000);
    } else {
      clearInterval(recTimerRef.current);
    }
    return () => clearInterval(recTimerRef.current);
  }, [isRecording]);

  // Loop playback within selected region
  const loopDebounceRef = useRef(0);
  useEffect(() => {
    if (!waveSelection || !isPlaying || audioDuration <= 0) return;
    const selEndSec = waveSelection.end * audioDuration;
    if (playbackPositionSecs >= selEndSec - 0.05) {
      const now = Date.now();
      if (now - loopDebounceRef.current > 400) {
        loopDebounceRef.current = now;
        seekAudio(waveSelection.start * audioDuration);
      }
    }
  }, [playbackPositionSecs]);

  // Clear selection when a new recording starts
  useEffect(() => {
    if (isRecording) setWaveSelection(null);
  }, [isRecording]);

  const firstFrogInSelection = useMemo(() => {
    if (!waveSelection || !waveformFrogLabels?.length) return null;
    const n     = waveformFrogLabels.length;
    const start = Math.floor(waveSelection.start * n);
    const end   = Math.ceil(waveSelection.end * n);
    for (let i = start; i <= end && i < n; i++) {
      if (waveformFrogLabels[i]) return waveformFrogLabels[i];
    }
    return null;
  }, [waveSelection, waveformFrogLabels]);

  if (!fontsLoaded) return null;

  const isFrog = prediction && prediction.label !== 'Background';
  const top    = prediction?.all?.[0] ?? (prediction ? { label: prediction.label, confidence: prediction.confidence } : null);
  const fmtSec = s => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;


  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
      <StatusBar barStyle="dark-content" backgroundColor={C.bg} />

      {/* Fixed header + tabs */}
      <View style={styles.topSection}>
        <Text style={[T.label, { color: C.textLight, textAlign: 'center', fontSize: 10 }]}>BIOACOUSTICS</Text>
        <Text style={[T.h1, { color: C.text, textAlign: 'center', marginTop: 2 }]}>FrogFinder</Text>
        <View style={styles.tabRow}>
          {TABS.map(tab => {
            const active = activePage === tab.page;
            return (
              <TouchableOpacity
                key={tab.page}
                onPress={() => goToPage(tab.page)}
                activeOpacity={0.75}
                style={[styles.tabPill, active && styles.tabPillActive]}
              >
                <Text style={[T.medium, {
                  fontSize: 13,
                  color: active ? C.pillText : C.textMid,
                  fontFamily: active ? 'Inter-SemiBold' : 'Inter-Regular',
                }]}>
                  {tab.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {/* Pages */}
      <ScrollView
        ref={scrollRef}
        horizontal
        pagingEnabled
        scrollEnabled={!waveGestureActive}
        showsHorizontalScrollIndicator={false}
        scrollEventThrottle={8}
        onMomentumScrollEnd={e => setActivePage(Math.round(e.nativeEvent.contentOffset.x / SW))}
        style={{ flex: 1 }}
      >

          {/* Page 0: About */}
          <View style={styles.page}>
            <ScrollView contentContainerStyle={{ padding: 20, gap: 14, paddingBottom: 40 }} showsVerticalScrollIndicator={false}>
<View style={styles.card}>
                <Text style={[T.label, { color: C.green, marginBottom: 10 }]}>PROJECT</Text>
                <Text style={[T.body, { color: C.textMid }]}>
                  FrogFinder uses on-device machine learning to identify frog species from their calls — no internet required.
                  By crowdsourcing detections, we help researchers track frog populations and detect early signs of habitat stress.
                </Text>
              </View>
              <View style={styles.card}>
                <Text style={[T.label, { color: C.green, marginBottom: 12 }]}>TEAM</Text>
                {TEAM.map(member => (
                  <View key={member.name} style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 12 }}>
                    <View style={styles.avatar}>
                      <Text style={[T.title, { color: C.green }]}>{member.initials}</Text>
                    </View>
                    <View style={{ marginLeft: 12 }}>
                      <Text style={[T.title, { color: C.text }]}>{member.name}</Text>
                      <Text style={[T.caption, { color: C.textMid }]}>{member.role}</Text>
                    </View>
                  </View>
                ))}
              </View>
            </ScrollView>
          </View>

          {/* Page 1: Record */}
          <View style={styles.page}>
            {(isRecording || isProcessing || !!prediction || pendingRecord) ? (
              /* Cards */
              <ScrollView contentContainerStyle={{ padding: 20, gap: 12 }} showsVerticalScrollIndicator={false}>

                {/* Card 1: recording / processing / analysed header */}
                <View style={styles.card}>
                  {isRecording && (<>
                    <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                        <View style={styles.recDot} />
                        <Text style={[T.title, { color: C.text }]}>Recording...</Text>
                      </View>
                      <Text style={[T.caption, { color: C.textMid, fontVariant: ['tabular-nums'] }]}>{fmtSec(recSeconds)}</Text>
                    </View>
                    <Waveform isActive={true} audioLevel={audioLevel} />
                  </>)}

                  {isProcessing && (<>
                    <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
                        <Ionicons name="analytics-outline" size={18} color={C.green} />
                        <Text style={[T.title, { color: C.text }]}>Analysing audio</Text>
                      </View>
                      {progress !== null && (
                        <Text style={[T.caption, { color: C.textMid }]}>{Math.round(progress * 100)}%</Text>
                      )}
                    </View>
                    <StatusTicker text={status || 'Starting...'} />
                    {waveformSamples.length > 0 && (
                      <View style={{ flexDirection: 'row', alignItems: 'flex-end', height: 36, gap: 1.5, marginTop: 14 }}>
                        {waveformSamples.map((amp, i) => {
                          const analysed = progress !== null && i < progress * waveformSamples.length;
                          return (
                            <View key={i} style={{
                              flex: 1, height: Math.max(2, amp * 36),
                              borderRadius: 1, backgroundColor: analysed ? C.green : C.border,
                            }} />
                          );
                        })}
                      </View>
                    )}
                    {progress !== null && (
                      <Animated.View style={{ opacity: progressFadeAnim, marginTop: 10 }}>
                        <View style={styles.confTrack}>
                          <Animated.View style={[styles.confFill, {
                            width: CARD_INNER_W,
                            transform: [{ translateX: progressAnim.interpolate({ inputRange: [0, 1], outputRange: [-CARD_INNER_W, 0] }) }],
                          }]} />
                        </View>
                      </Animated.View>
                    )}
                  </>)}

                  {!isRecording && !isProcessing && prediction && (
                    <Animated.View style={{ opacity: resultDropAnim }}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                          <View style={styles.micCircle}>
                            <Ionicons name="mic-outline" size={13} color={C.textMid} />
                          </View>
                          <View>
                            <Text style={[T.caption, { color: C.textMid, fontStyle: 'italic', fontSize: 12 }]}>Recording analysed</Text>
                            {audioDuration > 0 && (
                              <Text style={[T.caption, { color: C.textLight, fontSize: 11 }]}>{audioDuration}s · just now</Text>
                            )}
                          </View>
                        </View>
                        {firstFrogInSelection && (
                          <View style={{ alignItems: 'flex-end', maxWidth: 120 }}>
                            <Text style={[T.label, { color: C.textLight, fontSize: 9 }]}>IN SELECTION</Text>
                            <Text style={[T.caption, { color: C.green, fontSize: 11, fontFamily: 'Inter-Medium' }]} numberOfLines={1}>
                              {getNames(firstFrogInSelection).displayName}
                            </Text>
                          </View>
                        )}
                      </View>
                      {waveformSamples.length > 0 && (
                        <View style={{ marginTop: 12 }}>
                          <SelectionWaveform
                            samples={waveformSamples}
                            frogBuckets={waveformFrogBuckets}
                            playbackFraction={audioDuration > 0 ? playbackPositionSecs / audioDuration : 0}
                            selection={waveSelection}
                            onSelectionChange={setWaveSelection}
                            onGestureActiveChange={setWaveGestureActive}
                          />
                          <Text style={[T.label, { color: C.textLight, fontSize: 9, marginTop: 3 }]}>
                            {waveSelection ? 'LOOPING · TAP TO CLEAR' : 'DRAG TO SELECT LOOP REGION · GREEN = FROG'}
                          </Text>
                          <ScrubBar
                            value={audioDuration > 0 ? playbackPositionSecs / audioDuration : 0}
                            totalSecs={audioDuration}
                            onSeek={frac => seekAudio(frac * audioDuration)}
                            onGestureActiveChange={setWaveGestureActive}
                          />
                        </View>
                      )}
                    </Animated.View>
                  )}
                </View>

                {/* Card 2: species classification */}
                {!isRecording && !isProcessing && prediction && (() => {
                  const pct = Math.round(top.confidence * 100);
                  const { displayName, latin } = getNames(top.label);
                  const f1 = getModelF1(top.label, selectedModel);
                  return (
                    <Animated.View style={{ opacity: resultDropAnim }}>
                      <View style={styles.card}>
                        {isFrog && top ? (<>
                          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: 12 }}>
                            <View style={styles.frogThumb}>
                              <Text style={{ fontSize: 36 }}>🐸</Text>
                            </View>
                            <View style={{ flex: 1 }}>
                              <View style={{ flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                                <Text style={[T.h3, { color: C.text, flex: 1, marginRight: 8 }]} numberOfLines={2}>{displayName}</Text>
                                <Text style={[T.h3, { color: C.green }]}>{pct}%</Text>
                              </View>
                              {latin && <Text style={[T.caption, { color: C.textMid, fontStyle: 'italic', marginTop: 2 }]}>{latin}</Text>}
                            </View>
                          </View>
                          {f1 !== null && f1 > 0 && (
                            <Text style={[T.caption, { color: C.green, marginTop: 3 }]}>Model F1: {Math.round(f1 * 100)}%</Text>
                          )}
                          <View style={[styles.confTrack, { marginTop: 10 }]}>
                            <View style={[styles.confFill, { width: `${pct}%` }]} />
                          </View>
                          {(prediction.all?.length ?? 0) > 1 && (<>
                            {expandedResults && prediction.all.slice(1, 5).map(item => {
                              const p = Math.round(item.confidence * 100);
                              const { displayName: dn, latin: lt } = getNames(item.label);
                              const f = getModelF1(item.label, selectedModel);
                              return (
                                <View key={item.label}>
                                  <View style={styles.divider} />
                                  <View style={{ flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                                    <Text style={[T.h3, { color: C.text, flex: 1, marginRight: 10, fontSize: 18 }]} numberOfLines={1}>{dn}</Text>
                                    <Text style={[T.title, { color: C.amber }]}>{p}%</Text>
                                  </View>
                                  {lt && <Text style={[T.caption, { color: C.textMid, fontStyle: 'italic', marginTop: 2 }]}>{lt}</Text>}
                                  {f !== null && f > 0 && (
                                    <Text style={[T.caption, { color: C.green, marginTop: 2 }]}>Model F1: {Math.round(f * 100)}%</Text>
                                  )}
                                  <View style={[styles.confTrack, { marginTop: 6 }]}>
                                    <View style={[styles.confFill, { width: `${p}%`, backgroundColor: C.amber }]} />
                                  </View>
                                </View>
                              );
                            })}
                            <TouchableOpacity
                              onPress={() => setExpandedResults(e => !e)}
                              activeOpacity={0.7}
                              style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, paddingTop: 12 }}
                            >
                              <Text style={[T.caption, { color: C.textLight }]}>
                                {expandedResults ? 'Hide alternatives' : `${Math.min(prediction.all.length - 1, 4)} alternative${prediction.all.length > 2 ? 's' : ''}`}
                              </Text>
                              <Ionicons name={expandedResults ? 'chevron-up' : 'chevron-down'} size={13} color={C.textLight} />
                            </TouchableOpacity>
                          </>)}
                        </>) : (
                          <Text style={[T.title, { color: C.textMid }]}>No species detected</Text>
                        )}
                      </View>
                    </Animated.View>
                  );
                })()}

                {status && !isProcessing ? <Text style={[T.caption, { color: C.textLight, textAlign: 'center' }]}>{status}</Text> : null}
              </ScrollView>
            ) : (
              /* Idle */
              <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 40 }}>
                <Text style={{ fontSize: 72, marginBottom: 20 }}>🐸</Text>
                <Text style={[T.h3, { color: C.text, textAlign: 'center' }]}>Identify a frog</Text>
                <Text style={[T.body, { color: C.textMid, textAlign: 'center', marginTop: 10 }]}>
                  Tap Record and point your microphone toward frog calls, or upload an audio file.
                </Text>
              </View>
            )}
          </View>

          {/* Page 2: Sightings */}
          <View style={styles.page}>
            <ScrollView contentContainerStyle={{ padding: 20, paddingBottom: 40 }} showsVerticalScrollIndicator={false}>
              <Text style={[T.label, { color: C.textLight, marginBottom: 16 }]}>RECENT DETECTIONS</Text>
              {SIGHTINGS.map((s, i) => (
                <View key={i} style={[styles.sightingRow, i < SIGHTINGS.length - 1 && styles.sightingDivider]}>
                  <View style={{ flex: 1, marginRight: 12 }}>
                    <Text style={[T.h3, { color: C.text, fontSize: 17 }]}>{s.species}</Text>
                    <Text style={[T.caption, { color: C.textMid, marginTop: 3 }]}>{s.location} · {s.date}</Text>
                  </View>
                  <View style={styles.badge}>
                    <Text style={[T.label, { color: C.green, fontSize: 11 }]}>{s.confidence}%</Text>
                  </View>
                </View>
              ))}
            </ScrollView>
          </View>

      </ScrollView>

      {/* Bottom action bar */}
      <SafeAreaView edges={['bottom']} style={{ backgroundColor: C.surface }}>
        <View style={styles.bottomBar}>
          <View style={{ alignItems: 'center', paddingBottom: 10 }}>
            <View style={styles.modelChip}>
              <Ionicons name="sunny-outline" size={13} color={C.green} />
              <Text style={[T.caption, { color: C.text, marginLeft: 6, marginRight: 4, fontSize: 12, fontFamily: 'Inter-Medium' }]}>
                {selectedModel}
              </Text>
              <Ionicons name="chevron-down" size={13} color={C.textMid} />
            </View>
          </View>

          <View style={styles.actionRow}>
            <TouchableOpacity onPress={uploadAudio} disabled={isRecording || isProcessing} activeOpacity={0.7}
              style={[styles.sideAction, (isRecording || isProcessing) && { opacity: 0.35 }]}>
              <Ionicons name="arrow-up-outline" size={22} color={C.textMid} />
              <Text style={[T.label, { color: C.textMid, marginTop: 3, fontSize: 10 }]}>UPLOAD</Text>
            </TouchableOpacity>

            <TouchableOpacity
              onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Rigid);
                if (isProcessing) { cancel(); }
                else if (isRecording) { stopRecording(); }
                else {
                  const onRecordTab = activePage === 1;
                  if (!onRecordTab) goToPage(1);
                  const doStart = () => {
                    LayoutAnimation.configureNext({
                      duration: 280,
                      create: { type: LayoutAnimation.Types.easeInEaseOut, property: LayoutAnimation.Properties.opacity },
                      update: { type: LayoutAnimation.Types.easeInEaseOut },
                    });
                    setPendingRecord(true);
                    startRecording();
                  };
                  if (onRecordTab) doStart();
                  else setTimeout(doStart, 340);
                  return; // haptics already called above
                }
              }}
              activeOpacity={0.85}
              style={[styles.recordPill, isRecording && styles.recordPillStop]}
            >
              <Ionicons name={(isRecording || isProcessing) ? 'stop' : 'mic'} size={20} color={C.pillText} />
              <Text style={[T.title, { color: C.pillText, marginLeft: 9, fontSize: 16 }]}>
                {isProcessing ? 'Cancel analysis' : isRecording ? 'Stop recording' : 'Record'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              onPress={() => { if (audioUri) { setWaveSelection(null); playAudio(); } }}
              disabled={!audioUri || isProcessing} activeOpacity={0.7}
              style={[styles.sideAction, (!audioUri || isProcessing) && { opacity: 0.35 }]}>
              <Ionicons name={isPlaying ? 'pause' : 'play'} size={22} color={C.textMid} />
              <Text style={[T.label, { color: C.textMid, marginTop: 3, fontSize: 10 }]}>PLAY</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <AppContent />
    </SafeAreaProvider>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------
const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: C.bg },
  page:     { width: SW, height: '100%' },

  frogThumb: {
    width: 64, height: 64, borderRadius: 14,
    borderWidth: 3, borderColor: '#FFFFFF',
    backgroundColor: C.greenLight,
    alignItems: 'center', justifyContent: 'center',
  },

  topSection: {
    paddingTop: 14, paddingBottom: 12, paddingHorizontal: 24,
    backgroundColor: C.bg,
  },
  tabRow: {
    flexDirection: 'row', alignSelf: 'center',
    backgroundColor: C.surface, borderRadius: 99,
    padding: 4, marginTop: 14,
    borderWidth: 1, borderColor: C.border,
  },
  tabPill:       { paddingHorizontal: 18, paddingVertical: 8, borderRadius: 99 },
  tabPillActive: { backgroundColor: C.pill },

  card: {
    backgroundColor: C.surface, borderRadius: 16, padding: 16,
    shadowColor: '#6B4F2A', shadowOffset: { width: 0, height: 2 },
    shadowRadius: 10, shadowOpacity: 0.10, elevation: 3,
  },
  cardHeaderRow: {
    paddingBottom: 10,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  micCircle: {
    width: 28, height: 28, borderRadius: 14,
    backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center',
  },

  recDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: C.red },

  divider:   { height: 1, backgroundColor: C.border, marginVertical: 12 },
  confTrack: { height: 4, backgroundColor: C.border, borderRadius: 99, overflow: 'hidden' },
  confFill:  { height: '100%', backgroundColor: C.green, borderRadius: 99 },

  bottomBar: {
    paddingHorizontal: 16, paddingTop: 12, paddingBottom: 6,
    borderTopWidth: 1, borderTopColor: C.border,
    backgroundColor: C.surface,
  },
  modelChip: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: C.bg, borderRadius: 99,
    paddingHorizontal: 14, paddingVertical: 7,
    borderWidth: 1, borderColor: C.border,
  },
  actionRow:  { flexDirection: 'row', alignItems: 'center', gap: 8 },
  sideAction: { width: 60, alignItems: 'center', justifyContent: 'center', paddingVertical: 6 },
  recordPill: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.pill, borderRadius: 99, paddingVertical: 15,
  },
  recordPillStop: { backgroundColor: C.red },

  avatar:          { width: 40, height: 40, borderRadius: 20, backgroundColor: C.greenLight, alignItems: 'center', justifyContent: 'center' },
  badge:           { backgroundColor: C.greenLight, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3 },
  sightingRow:     { paddingVertical: 14 },
  sightingDivider: { borderBottomWidth: 1, borderBottomColor: C.border },
});
