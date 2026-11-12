package main

import (
	"github.com/yoonsung9948/relay/internal/config"
	"io"
	"testing"
)

func TestConfigFlag(t *testing.T) {
	for _, tc := range []struct {
		args    []string
		want    string
		invalid bool
	}{
		{nil, config.ConfigPath, false},
		{[]string{"--config", "/tmp/control.yaml"}, "/tmp/control.yaml", false},
		{[]string{"--path", "/app/docker/pool.yaml"}, "", true},
		{[]string{"unexpected"}, "", true},
	} {
		got, err := configPath(tc.args, io.Discard)
		if (err != nil) != tc.invalid || got != tc.want {
			t.Fatalf("args=%v got=%q err=%v", tc.args, got, err)
		}
	}
}
