package metrics

import "time"

type RequestMetrics struct {
	RequestID string

	StartTime      time.Time
	FirstTokenTime time.Time
	LastTokenTime  time.Time
	EndTime        time.Time

	TokenTimestamps []time.Time

	InputTokens  int
	OutputTokens int

	Error error
}

func (m RequestMetrics) TTFT() time.Duration {
	return m.FirstTokenTime.Sub(m.StartTime)
}

func (m RequestMetrics) E2E() time.Duration {
	return m.EndTime.Sub(m.StartTime)
}

func (m RequestMetrics) ITLs() []time.Duration {
	result := make([]time.Duration, 0)

	for i := 1; i < len(m.TokenTimestamps); i++ {
		delta := m.TokenTimestamps[i].Sub(
			m.TokenTimestamps[i-1],
		)

		result = append(result, delta)
	}

	return result
}
