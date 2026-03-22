import "./global.css";
import { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, Animated, StatusBar, StyleSheet, ScrollView, Dimensions } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useFonts, Nunito_400Regular, Nunito_600SemiBold, Nunito_700Bold, Nunito_800ExtraBold } from '@expo-google-fonts/nunito';

import { C, T } from './src/theme';
import { TEAM, SIGHTINGS } from './src/constants';
import { getNames, getModelF1, MODEL_NAMES } from './src/classifier';
import { useAudioProcessor } from './src/useAudioProcessor';
import { FlatCloud } from './src/FlatCloud';

const { width: SW, height: SH } = Dimensions.get('window');
const TRACK_W = SW - 40;

// ---------------------------------------------------------------------------
// AppContent — UI only, all logic lives in useAudioProcessor
// ---------------------------------------------------------------------------
function AppContent() {
  const [fontsLoaded] = useFonts({
    'Nunito-Regular':   Nunito_400Regular,
    'Nunito-SemiBold':  Nunito_600SemiBold,
    'Nunito-Bold':      Nunito_700Bold,
    'Nunito-ExtraBold': Nunito_800ExtraBold,
  });

  const {
    prediction, status, progress, audioDuration, isProcessing,
    expandedResults, setExpandedResults, isRecording,
    audioUri, isPlaying,
    progressAnim, progressFadeAnim, resultDropAnim,
    startRecording, stopRecording, uploadAudio, playAudio, cancel,
  } = useAudioProcessor(selectedModel);

  const [activePage,     setActivePage]     = useState(0);
  const [selectedModel,  setSelectedModel]  = useState(MODEL_NAMES[0]);
  const pageAnim    = useRef(new Animated.Value(0)).current;
  const tabPillAnim = useRef(new Animated.Value(0)).current;
  const pressAnim   = useRef(new Animated.Value(0)).current;
  const cloud1      = useRef(new Animated.Value(SW + 80)).current;
  const cloud2      = useRef(new Animated.Value(SW + 80)).current;
  const cloud3      = useRef(new Animated.Value(SW + 80)).current;
  const shakeAnim       = useRef(new Animated.Value(0)).current;
  const heroCrossAnim   = useRef(new Animated.Value(1)).current;
  const textSlideAnim   = useRef(new Animated.Value(300)).current;
  const labelSlideAnim  = useRef(new Animated.Value(300)).current;
  const isMounted       = useRef(false);
  const pulse1  = useRef(new Animated.Value(1)).current;
  const pulse2  = useRef(new Animated.Value(1)).current;
  const opacity1 = useRef(new Animated.Value(0)).current;
  const opacity2 = useRef(new Animated.Value(0)).current;
  const loop1 = useRef(null);
  const loop2 = useRef(null);

  const pressIn  = () => Animated.spring(pressAnim, { toValue: 4, useNativeDriver: true, speed: 60, bounciness: 0 }).start();
  const pressOut = () => Animated.spring(pressAnim, { toValue: 0, useNativeDriver: true, speed: 20, bounciness: 6 }).start();

  const goToPage = (page) => {
    setActivePage(page);
    Animated.timing(pageAnim,    { toValue: -page * SW, duration: 280, useNativeDriver: true }).start();
    Animated.spring(tabPillAnim, { toValue: page, useNativeDriver: true, speed: 14, bounciness: 12 }).start();
  };

  // Recording pulse rings
  useEffect(() => {
    if (isRecording) {
      const makeLoop = (scale, opacity, delay) => {
        scale.setValue(1); opacity.setValue(0.6);
        return Animated.loop(Animated.sequence([
          Animated.delay(delay),
          Animated.parallel([
            Animated.timing(scale,   { toValue: 2.8, duration: 1600, useNativeDriver: true }),
            Animated.timing(opacity, { toValue: 0,   duration: 1600, useNativeDriver: true }),
          ]),
          Animated.parallel([
            Animated.timing(scale,   { toValue: 1, duration: 0, useNativeDriver: true }),
            Animated.timing(opacity, { toValue: 0.6, duration: 0, useNativeDriver: true }),
          ]),
        ]));
      };
      loop1.current = makeLoop(pulse1, opacity1, 0);
      loop2.current = makeLoop(pulse2, opacity2, 700);
      loop1.current.start();
      loop2.current.start();
    } else {
      loop1.current?.stop(); loop2.current?.stop();
      opacity1.setValue(0); opacity2.setValue(0);
      pulse1.setValue(1);   pulse2.setValue(1);
    }
  }, [isRecording]);

  // Drifting clouds
  useEffect(() => {
    const startCloud = (anim, duration, delay) => {
      anim.setValue(SW + 80);
      Animated.loop(Animated.sequence([
        Animated.delay(delay),
        Animated.timing(anim, { toValue: -160, duration, useNativeDriver: true }),
        Animated.timing(anim, { toValue: SW + 80, duration: 0, useNativeDriver: true }),
      ])).start();
    };
    startCloud(cloud1, 20000,     0);
    startCloud(cloud2, 28000,  8000);
    startCloud(cloud3, 16000, 14000);
  }, []);

  // Crossfade hero on idle ↔ recording transition (skip when entering processing)
  useEffect(() => {
    if (!isMounted.current) { isMounted.current = true; return; }
    if (isProcessing) return;
    heroCrossAnim.setValue(0.15);
    Animated.timing(heroCrossAnim, { toValue: 1, duration: 280, useNativeDriver: true }).start();
  }, [isProcessing, isRecording]);

  // Periodic frog icon shake
  useEffect(() => {
    let timeout;
    const scheduleShake = () => {
      timeout = setTimeout(() => {
        Animated.sequence([
          Animated.timing(shakeAnim, { toValue: -9, duration: 55, useNativeDriver: true }),
          Animated.timing(shakeAnim, { toValue:  9, duration: 55, useNativeDriver: true }),
          Animated.timing(shakeAnim, { toValue: -6, duration: 55, useNativeDriver: true }),
          Animated.timing(shakeAnim, { toValue:  6, duration: 55, useNativeDriver: true }),
          Animated.timing(shakeAnim, { toValue:  0, duration: 55, useNativeDriver: true }),
        ]).start(() => scheduleShake());
      }, 3500 + Math.random() * 3000);
    };
    scheduleShake();
    return () => clearTimeout(timeout);
  }, []);

  // Slide dialogue text in from right when prediction changes
  useEffect(() => {
    textSlideAnim.setValue(300); labelSlideAnim.setValue(300);
    Animated.parallel([
      Animated.spring(textSlideAnim,  { toValue: 0, useNativeDriver: true, speed: 16, bounciness: 3 }),
      Animated.sequence([
        Animated.delay(90),
        Animated.spring(labelSlideAnim, { toValue: 0, useNativeDriver: true, speed: 16, bounciness: 3 }),
      ]),
    ]).start();
  }, [prediction]);

  if (!fontsLoaded) return null;

  const isFrog = prediction && prediction.label !== 'Background';
  const busy   = isProcessing || isRecording;

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
      <StatusBar barStyle="dark-content" backgroundColor={C.bg} />

      <View style={{ flex: 1, overflow: 'hidden' }}>
        <Animated.View style={[styles.pagesRow, { transform: [{ translateX: pageAnim }] }]}>

          {/* ── PAGE 1: Classifier ── */}
          <View style={styles.page}>
            {/* Drifting clouds */}
            <Animated.View style={[styles.cloud, { top: SH * 0.09, transform: [{ translateX: cloud1 }] }]}><FlatCloud width={120} opacity={0.85} /></Animated.View>
            <Animated.View style={[styles.cloud, { top: SH * 0.18, transform: [{ translateX: cloud2 }] }]}><FlatCloud width={80}  opacity={0.70} /></Animated.View>
            <Animated.View style={[styles.cloud, { top: SH * 0.04, transform: [{ translateX: cloud3 }] }]}><FlatCloud width={100} opacity={0.78} /></Animated.View>

            {/* Header */}
            <View style={{ paddingHorizontal: 24, paddingTop: 8, paddingBottom: 10 }}>
              {/* Mascot — absolutely positioned, no layout impact */}
              <View style={{ position: 'absolute', top: 8, right: 24, zIndex: 2 }}>
                <View style={styles.mascotCircle}>
                  <Text style={{ fontSize: 36 }}>🐸</Text>
                </View>
              </View>

              {/* Field guide badge */}
              <View style={{ alignSelf: 'flex-start' }}>
                <View style={[styles.fieldGuideBadge, { position: 'absolute', top: 4, backgroundColor: '#b89a10' }]} />
                <View style={styles.fieldGuideBadge}>
                  <Text style={{ fontSize: 14 }}>🐸</Text>
                  <Text style={[T.label, { color: C.brown, marginLeft: 6 }]}>YOUR FIELD GUIDE</Text>
                </View>
              </View>

              <Text style={[T.h1, { color: C.brown, marginTop: 3, paddingRight: 88, textShadowColor: 'rgba(61,50,38,0.22)', textShadowOffset: { width: 2, height: 4 }, textShadowRadius: 1 }]}>
                FrogFinder
              </Text>

              {/* Dialogue box — Ribbit label overlaps top-left */}
              <View style={{ marginTop: 14 }}>
                <Animated.View style={[styles.ribbitLabel, { transform: [{ translateX: labelSlideAnim }], zIndex: 2 }]}>
                  <Text style={[T.label, { color: C.brown }]}>Ribbit! 🐸</Text>
                </Animated.View>
                <View style={[styles.acDialogue, { position: 'absolute', bottom: 3, left: 0, right: 0, backgroundColor: C.brownLight, zIndex: 0 }]} />
                <View style={[styles.acDialogue, { zIndex: 1 }]}>
                  <View style={{ overflow: 'hidden' }}>
                    <Animated.View style={{ transform: [{ translateX: textSlideAnim }] }}>
                      <Text style={[T.body, { color: C.brownMid, fontSize: 15, lineHeight: 24 }]}>
                        {prediction
                          ? (isFrog
                              ? <>{'Wowie!! I found a friend out there~ 🌿\nIt sounds like a '}<Text style={{ fontFamily: 'Nunito-ExtraBold', color: C.green }}>{getNames(prediction.label).displayName}</Text>{'!'}</>
                              : "Hmm... I couldn't hear any frogs nearby! 🤔\nTry getting a little closer~")
                          : "Point your phone at frog calls to discover who's singing!"
                        }
                      </Text>
                    </Animated.View>
                  </View>
                </View>
              </View>
            </View>

            {/* Hero — flex:1 so it takes remaining space above the bottom panel */}
            <Animated.View style={[styles.hero, { opacity: heroCrossAnim }]}>
              {isProcessing ? (
                <View style={{ alignItems: 'center', gap: 16 }}>
                  <View style={{ position: 'relative' }}>
                    <View style={[styles.recordBtn, styles.recordBtnShadowCancel, { position: 'absolute', top: 4 }]} />
                    <Animated.View style={{ transform: [{ translateY: pressAnim }] }}>
                      <TouchableOpacity
                        onPressIn={pressIn} onPressOut={pressOut}
                        onPress={cancel}
                        activeOpacity={1}
                        style={[styles.recordBtn, styles.recordBtnCancel]}
                      >
                        <Ionicons name="close" size={40} color={C.brown} />
                      </TouchableOpacity>
                    </Animated.View>
                  </View>
                  <Text style={[T.medium, { color: C.brownMid }]}>Stop classifying</Text>
                </View>
              ) : (
                <View style={{ alignItems: 'center' }}>
                  <View style={styles.btnWrapper}>
                    <Animated.View style={[styles.ring, { transform: [{ scale: pulse1 }], opacity: opacity1 }]} />
                    <Animated.View style={[styles.ring, { transform: [{ scale: pulse2 }], opacity: opacity2 }]} />
                    <View style={{ position: 'relative' }}>
                      <View style={[styles.recordBtn,
                        isRecording ? styles.recordBtnShadowActive : styles.recordBtnShadowIdle,
                        { position: 'absolute', top: 4 }
                      ]} />
                      <Animated.View style={{ transform: [{ translateY: pressAnim }] }}>
                        <TouchableOpacity
                          onPressIn={pressIn} onPressOut={pressOut}
                          onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Rigid);
                            isRecording ? stopRecording() : startRecording();
                          }}
                          activeOpacity={1}
                          style={[styles.recordBtn, isRecording ? styles.recordBtnActive : styles.recordBtnIdle]}
                        >
                          <Ionicons name={isRecording ? 'stop' : 'mic'} size={40} color="white" />
                        </TouchableOpacity>
                      </Animated.View>
                    </View>
                  </View>
                  <Text style={[T.title, { color: C.brownMid, marginTop: 24 }]}>
                    {isRecording ? 'Listening for frogs...' : 'Tap to listen!'}
                  </Text>
                </View>
              )}
            </Animated.View>

            {/* Bottom panel — normal flow so hero flex:1 adjusts naturally */}
            <View style={styles.bottomPanel}>
              <ScrollView
                style={{ maxHeight: SH * 0.33 }}
                contentContainerStyle={{ paddingHorizontal: 20, paddingTop: 12, paddingBottom: 8, gap: 10 }}
                showsVerticalScrollIndicator={false}
              >
                {/* Progress bar */}
                {progress !== null && (
                  <Animated.View style={{
                    opacity: progressFadeAnim,
                    transform: [{ translateY: progressFadeAnim.interpolate({ inputRange: [0, 1], outputRange: [32, 0] }) }],
                  }}>
                    <Text style={[T.caption, { color: C.brownMid, marginBottom: 8 }]}>
                      Searching... {Math.round(progress * 100)}%
                    </Text>
                    <View style={styles.confTrack}>
                      <Animated.View style={[styles.confFill, {
                        width: TRACK_W,
                        transform: [{ translateX: progressAnim.interpolate({ inputRange: [0, 1], outputRange: [-TRACK_W, 0] }) }],
                      }]} />
                    </View>
                  </Animated.View>
                )}

                {/* Result card */}
                {prediction && (
                  <Animated.View style={{
                    opacity: resultDropAnim,
                    transform: [{ translateY: resultDropAnim.interpolate({ inputRange: [0, 1], outputRange: [50, 0] }) }],
                  }}>
                    <View style={styles.resultCard}>
                      <View style={[styles.resultCardHeader, !isFrog && { backgroundColor: C.redLight }]}>
                        <Text style={[T.title, { color: C.green }]}>
                          {isFrog ? '⭐  FOUND A FRIEND!' : '🤔  NO FROGS NEARBY'}
                        </Text>
                        {audioDuration > 0 && (
                          <View style={styles.durationPill}>
                            <Text style={[T.caption, { color: C.brownMid }]}>{audioDuration}s analysed</Text>
                          </View>
                        )}
                      </View>

                      {isFrog && (
                        <View style={{ padding: 16 }}>
                          {/* Winner row */}
                          {(() => {
                            const top = prediction.all?.[0] ?? { label: prediction.label, confidence: prediction.confidence };
                            const pct = Math.round(top.confidence * 100);
                            const { displayName, latin } = getNames(top.label);
                            const f1 = getModelF1(top.label, selectedModel);
                            return (
                              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 14 }}>
                                <Animated.View style={{ transform: [{ rotate: shakeAnim.interpolate({ inputRange: [-9, 0, 9], outputRange: ['-12deg', '0deg', '12deg'] }) }] }}>
                                  <View style={styles.frogSquareLarge}>
                                    <Text style={{ fontSize: 40 }}>🐸</Text>
                                  </View>
                                </Animated.View>
                                <View style={{ flex: 1 }}>
                                  <Text style={[T.h3, { color: C.brown }]}>{displayName}</Text>
                                  {latin && <Text style={[T.caption, { color: C.brown, fontStyle: 'italic', opacity: 0.55, marginTop: 2 }]}>{latin}</Text>}
                                  {f1 !== null && <Text style={[T.caption, { color: C.green, opacity: 0.8, marginTop: 2 }]}>Model accuracy: {Math.round(f1 * 100)}%</Text>}
                                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 8 }}>
                                    <View style={[styles.confTrack, { flex: 1 }]}>
                                      <View style={[styles.confFill, { width: `${pct}%` }]} />
                                    </View>
                                    <Text style={[T.title, { color: C.green, minWidth: 42, textAlign: 'right' }]}>{pct}%</Text>
                                  </View>
                                </View>
                              </View>
                            );
                          })()}

                          {/* Expand/collapse other species */}
                          {prediction.all.length > 1 && (
                            <>
                              {expandedResults && (
                                <ScrollView style={{ maxHeight: 200, marginTop: 4 }} nestedScrollEnabled showsVerticalScrollIndicator={false}>
                                  {prediction.all.slice(1, 5).map((item, idx) => {
                                    const pct = Math.round(item.confidence * 100);
                                    const { displayName: dn, latin: lt } = getNames(item.label);
                                    return (
                                      <View key={item.label}>
                                        {idx === 0 && (
                                          <View style={styles.leafDivider}>
                                            <View style={styles.leafLine} />
                                            <Text style={{ fontSize: 16 }}>🌿</Text>
                                            <View style={styles.leafLine} />
                                          </View>
                                        )}
                                        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: 12 }}>
                                          <View style={styles.frogSquareSmall}>
                                            <Text style={{ fontSize: 28 }}>🐸</Text>
                                          </View>
                                          <View style={{ flex: 1 }}>
                                            <Text style={[T.title, { color: C.brown }]}>{dn}</Text>
                                            {lt && <Text style={[T.caption, { color: C.brown, fontStyle: 'italic', opacity: 0.55, marginTop: 2 }]}>{lt}</Text>}
                                            {(() => { const f1 = getModelF1(item.label, selectedModel); return f1 !== null ? <Text style={[T.caption, { color: C.green, opacity: 0.7, marginTop: 2 }]}>Model accuracy: {Math.round(f1 * 100)}%</Text> : null; })()}
                                            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 6 }}>
                                              <View style={[styles.confTrack, { flex: 1 }]}>
                                                <View style={[styles.confFill, { width: `${pct}%`, backgroundColor: C.yellow }]} />
                                              </View>
                                              <Text style={[T.title, { color: C.yellowDark, minWidth: 42, textAlign: 'right' }]}>{pct}%</Text>
                                            </View>
                                          </View>
                                        </View>
                                      </View>
                                    );
                                  })}
                                </ScrollView>
                              )}
                              <TouchableOpacity onPress={() => setExpandedResults(e => !e)} activeOpacity={0.7} style={styles.expandBtn}>
                                <Text style={[T.caption, { color: C.brownLight }]}>
                                  {expandedResults ? 'Hide others' : `${Math.min(prediction.all.length - 1, 4)} others nearby`}
                                </Text>
                                <Ionicons name={expandedResults ? 'chevron-up' : 'chevron-down'} size={14} color={C.brownLight} />
                              </TouchableOpacity>
                            </>
                          )}
                        </View>
                      )}
                    </View>
                  </Animated.View>
                )}

                {status ? (
                  <Text style={[T.caption, { color: C.brownMid, textAlign: 'center', opacity: 0.7 }]}>{status}</Text>
                ) : null}
              </ScrollView>

              {/* Upload / Play / Model — pinned at bottom of panel */}
              <View style={styles.uploadRow}>
                <View style={styles.dashedSeparator} />

                {/* Upload + Play row */}
                <View style={{ flexDirection: 'row', gap: 8 }}>
                  <TouchableOpacity
                    onPress={uploadAudio}
                    disabled={busy}
                    activeOpacity={0.7}
                    style={[styles.uploadBtn, { flex: 1 }, busy && styles.uploadBtnDisabled]}
                  >
                    <Text style={{ fontSize: 16 }}>☁️</Text>
                    <Text style={[T.medium, { color: C.brownMid, marginLeft: 8 }]}>Upload</Text>
                  </TouchableOpacity>

                  {audioUri && (
                    <TouchableOpacity
                      onPress={playAudio}
                      disabled={isProcessing}
                      activeOpacity={0.7}
                      style={[styles.uploadBtn, styles.playBtn, isProcessing && styles.uploadBtnDisabled]}
                    >
                      <Ionicons
                        name={isPlaying ? 'pause' : 'play'}
                        size={18}
                        color={C.green}
                      />
                      <Text style={[T.medium, { color: C.green, marginLeft: 6 }]}>
                        {isPlaying ? 'Pause' : 'Play'}
                      </Text>
                    </TouchableOpacity>
                  )}
                </View>

                {/* Model picker */}
                <View style={styles.modelRow}>
                  <Text style={[T.caption, { color: C.brownLight, marginRight: 8 }]}>Model</Text>
                  <View style={{ flexDirection: 'row', gap: 6, flexWrap: 'wrap' }}>
                    {MODEL_NAMES.map(name => {
                      const active = name === selectedModel;
                      return (
                        <TouchableOpacity
                          key={name}
                          onPress={() => setSelectedModel(name)}
                          disabled={busy}
                          activeOpacity={0.7}
                          style={[styles.modelPill, active && styles.modelPillActive]}
                        >
                          <Text style={[T.caption, { color: active ? C.green : C.brownLight, fontFamily: active ? 'Nunito-Bold' : 'Nunito-Regular' }]}>
                            {name}
                          </Text>
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                </View>
              </View>
            </View>
          </View>

          {/* ── PAGE 2: About ── */}
          <View style={styles.page}>
            <View style={{ paddingHorizontal: 24, paddingTop: 16, paddingBottom: 8 }}>
              <View style={[styles.fieldGuideBadge, { alignSelf: 'flex-start' }]}>
                <Text style={{ fontSize: 14 }}>🌿</Text>
                <Text style={[T.label, { color: C.brown, marginLeft: 6 }]}>LEARN MORE</Text>
              </View>
              <Text style={[T.h1, { color: C.brown, marginTop: 8 }]}>About</Text>
            </View>

            <ScrollView style={{ flex: 1 }} showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: 24, gap: 16 }}>
              <View style={styles.aboutCard}>
                <Text style={[T.label, { color: C.green, marginBottom: 8 }]}>OUR MISSION</Text>
                <Text style={[T.body, { color: C.brownMid }]}>
                  FrogFinder uses on-device machine learning to identify frog species from their calls — no internet required.
                  By crowdsourcing detections, we help researchers track frog populations and detect early signs of habitat stress.
                </Text>
              </View>

              <View style={styles.aboutCard}>
                <Text style={[T.label, { color: C.green, marginBottom: 12 }]}>THE TEAM</Text>
                {TEAM.map((member) => (
                  <View key={member.name} style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 12 }}>
                    <View style={styles.avatar}>
                      <Text style={[T.title, { color: C.green }]}>{member.initials}</Text>
                    </View>
                    <View style={{ marginLeft: 12 }}>
                      <Text style={[T.title, { color: C.brown }]}>{member.name}</Text>
                      <Text style={[T.caption, { color: C.brownMid }]}>{member.role}</Text>
                    </View>
                  </View>
                ))}
              </View>

              <View style={styles.aboutCard}>
                <Text style={[T.label, { color: C.green, marginBottom: 12 }]}>RECENT SIGHTINGS</Text>
                {SIGHTINGS.map((s, i) => (
                  <View key={i} style={[{ paddingBottom: 12 }, i < SIGHTINGS.length - 1 && styles.sightingDivider]}>
                    <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Text style={[T.medium, { color: C.brown, flex: 1, marginRight: 8 }]}>{s.species}</Text>
                      <View style={styles.badge}>
                        <Text style={[T.label, { color: C.green }]}>{s.confidence}%</Text>
                      </View>
                    </View>
                    <Text style={[T.caption, { color: C.brownMid, marginTop: 3 }]}>
                      {s.location} · {s.date}
                    </Text>
                  </View>
                ))}
              </View>
            </ScrollView>
          </View>

        </Animated.View>
      </View>

      {/* Tab bar — SafeAreaView fills home-indicator gap */}
      <SafeAreaView edges={['bottom']} style={{ backgroundColor: C.tabBg }}>
        <View style={styles.tabBar}>
          <Animated.View style={[styles.tabPill, {
            transform: [{ translateX: tabPillAnim.interpolate({ inputRange: [0, 1], outputRange: [4, SW / 2 + 4] }) }],
          }]} />
          {[
            { icon: 'leaf-outline',               label: 'Identify', page: 0 },
            { icon: 'information-circle-outline', label: 'About',    page: 1 },
          ].map(({ icon, label, page }) => {
            const active = activePage === page;
            return (
              <TouchableOpacity
                key={page}
                onPress={() => goToPage(page)}
                disabled={busy}
                style={styles.tabItem}
                activeOpacity={0.7}
              >
                <Ionicons name={icon} size={22} color={active ? C.green : C.brownLight} />
                <Text style={[T.caption, {
                  fontFamily: active ? 'Nunito-Bold' : 'Nunito-Regular',
                  fontSize: 11,
                  color: active ? C.green : C.brownLight,
                  marginTop: 3,
                }]}>
                  {label}
                </Text>
              </TouchableOpacity>
            );
          })}
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
  safeArea:    { flex: 1, backgroundColor: C.bg },
  pagesRow:    { position: 'absolute', top: 0, bottom: 0, left: 0, flexDirection: 'row', width: SW * 2 },
  page:        { width: SW, height: '100%' },
  hero:        { flex: 1, alignItems: 'center', justifyContent: 'center' },
  bottomPanel: { backgroundColor: C.surface, borderTopLeftRadius: 24, borderTopRightRadius: 24 },

  fieldGuideBadge: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: C.yellow, borderRadius: 99,
    paddingHorizontal: 14, paddingVertical: 7,
    shadowColor: C.brown, shadowOffset: { width: 0, height: 2 }, shadowRadius: 4, shadowOpacity: 0.15, elevation: 3,
  },
  mascotCircle: {
    width: 64, height: 64, borderRadius: 32,
    borderWidth: 3, borderColor: C.green,
    backgroundColor: C.greenLight,
    alignItems: 'center', justifyContent: 'center',
  },
  ribbitLabel: {
    alignSelf: 'flex-start',
    backgroundColor: C.white,
    borderRadius: 99,
    paddingHorizontal: 13, paddingVertical: 5,
    marginLeft: 12, marginBottom: -12, zIndex: 1,
    borderWidth: 1.5, borderColor: C.yellow,
    shadowColor: C.brown, shadowOffset: { width: 0, height: 2 }, shadowRadius: 4, shadowOpacity: 0.15, elevation: 3,
  },

  btnWrapper:  { width: 124, height: 128, alignItems: 'center', justifyContent: 'center' },
  ring: {
    position: 'absolute', width: 120, height: 120, borderRadius: 28,
    borderWidth: 2.5, borderColor: C.green,
  },
  recordBtn: {
    width: 120, height: 120, borderRadius: 28,
    alignItems: 'center', justifyContent: 'center',
    shadowOffset: { width: 0, height: 6 }, shadowRadius: 16, shadowOpacity: 0.25, elevation: 10,
  },
  recordBtnCancel:       { backgroundColor: C.tabBg,  shadowColor: C.brownLight },
  recordBtnIdle:         { backgroundColor: C.green,  shadowColor: C.green },
  recordBtnActive:       { backgroundColor: C.coral,  shadowColor: C.coral },
  recordBtnShadowIdle:   { backgroundColor: '#3d6b30' },
  recordBtnShadowActive: { backgroundColor: '#a05040' },
  recordBtnShadowCancel: { backgroundColor: C.brownLight },

  resultCard: {
    backgroundColor: C.surface,
    borderWidth: 1.5, borderColor: C.border,
    borderRadius: 22, overflow: 'hidden',
    shadowColor: C.brown, shadowOffset: { width: 0, height: 4 }, shadowRadius: 12, shadowOpacity: 0.1, elevation: 5,
  },
  resultCardHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: C.mint, paddingHorizontal: 16, paddingVertical: 12,
  },
  durationPill: {
    backgroundColor: C.white, borderRadius: 99,
    paddingHorizontal: 12, paddingVertical: 4,
    borderWidth: 1, borderColor: C.border,
  },
  frogSquareLarge: {
    width: 80, height: 80, borderRadius: 18,
    backgroundColor: C.greenLight, alignItems: 'center', justifyContent: 'center',
  },
  frogSquareSmall: {
    width: 62, height: 62, borderRadius: 14,
    backgroundColor: 'rgba(255,209,102,0.2)',
    borderWidth: 1.5, borderColor: 'rgba(196,168,130,0.3)',
    alignItems: 'center', justifyContent: 'center',
  },
  leafDivider:  { flexDirection: 'row', alignItems: 'center', gap: 10, marginVertical: 14 },
  leafLine:     { flex: 1, height: 1, backgroundColor: C.border },
  expandBtn:    { flexDirection: 'row', alignItems: 'center', gap: 6, paddingVertical: 8, justifyContent: 'center' },

  confTrack: { height: 8, backgroundColor: 'rgba(61,50,38,0.08)', borderRadius: 99, overflow: 'hidden' },
  confFill:  { height: '100%', backgroundColor: C.green, borderRadius: 99 },

  uploadRow: { paddingHorizontal: 20, paddingBottom: 20, paddingTop: 4, gap: 8 },
  dashedSeparator: {
    borderWidth: 1.5, borderColor: C.border, borderStyle: 'dashed',
    borderRadius: 1, marginBottom: 12,
  },
  uploadBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    paddingVertical: 13, borderRadius: 16,
    backgroundColor: C.surface, borderWidth: 1.5, borderColor: C.border,
  },
  uploadBtnDisabled: { opacity: 0.4 },
  playBtn: { flex: 0, paddingHorizontal: 18, borderColor: C.green },

  modelRow: { flexDirection: 'row', alignItems: 'center', paddingTop: 4 },
  modelPill: {
    paddingHorizontal: 12, paddingVertical: 5, borderRadius: 99,
    borderWidth: 1.5, borderColor: C.border, backgroundColor: C.surface,
  },
  modelPillActive: { borderColor: C.green, backgroundColor: C.greenLight },

  tabBar:  { flexDirection: 'row', borderTopWidth: 1, borderTopColor: C.border, backgroundColor: C.tabBg },
  tabItem: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 10, zIndex: 1 },
  tabPill: { position: 'absolute', top: 6, bottom: 6, width: SW / 2 - 8, borderRadius: 12, backgroundColor: C.greenLight },

  aboutCard: {
    backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, borderRadius: 20, padding: 18,
    shadowColor: C.brown, shadowOffset: { width: 0, height: 3 }, shadowRadius: 8, shadowOpacity: 0.06, elevation: 3,
  },
  avatar:          { width: 42, height: 42, borderRadius: 12, backgroundColor: C.greenLight, alignItems: 'center', justifyContent: 'center' },
  badge:           { backgroundColor: C.greenLight, borderRadius: 10, paddingHorizontal: 8, paddingVertical: 3 },
  sightingDivider: { borderBottomWidth: 1, borderBottomColor: C.border, marginBottom: 12 },

  acDialogue: {
    backgroundColor: C.surface,
    borderWidth: 2.5, borderColor: C.brownLight,
    borderRadius: 18, padding: 16,
    shadowColor: C.brownLight,
    shadowOffset: { width: 0, height: 4 }, shadowRadius: 0, shadowOpacity: 1,
    elevation: 6,
  },

  cloud: { position: 'absolute' },
});
