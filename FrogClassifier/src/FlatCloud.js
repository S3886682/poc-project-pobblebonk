import { View } from 'react-native';

// Flat AC-style cloud made from overlapping white rounded circles.
// Width scales all proportions; opacity lets each cloud have different alpha.
export function FlatCloud({ width = 100, opacity = 0.88 }) {
  const b = width * 0.18;
  return (
    <View style={{ width, height: b * 3.2, opacity }}>
      <View style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: b * 1.1, backgroundColor: 'white', borderRadius: b * 0.55 }} />
      <View style={{ position: 'absolute', bottom: b * 0.55, left: width * 0.08,  width: b * 2,   height: b * 2,   backgroundColor: 'white', borderRadius: 999 }} />
      <View style={{ position: 'absolute', bottom: b * 0.9,  left: width * 0.30,  width: b * 2.6, height: b * 2.6, backgroundColor: 'white', borderRadius: 999 }} />
      <View style={{ position: 'absolute', bottom: b * 0.55, right: width * 0.10, width: b * 1.7, height: b * 1.7, backgroundColor: 'white', borderRadius: 999 }} />
    </View>
  );
}
