package controlplane

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"syscall"
	"time"
)

// WorkerRecord is the sole ownership record. It contains no prompts or credentials.
type WorkerRecord struct {
	AttemptID    string    `json:"attempt_id"`
	Name         string    `json:"name"`
	Provider     string    `json:"provider"`
	Scope        string    `json:"scope"`
	InstanceID   string    `json:"instance_id,omitempty"`
	Phase        string    `json:"phase"`
	CreatedAt    time.Time `json:"created_at"`
	ReadyAt      time.Time `json:"ready_at,omitempty"`
	LastActivity time.Time `json:"last_activity,omitempty"`
}
type journalData struct {
	Version int           `json:"version"`
	OwnerID string        `json:"owner_id"`
	Worker  *WorkerRecord `json:"worker"`
}
type ownershipJournal struct {
	data      journalData
	directory string
	lock      *os.File
	closed    bool
}

func uniqueID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(fmt.Sprintf("generate ownership identity: %v", err))
	}
	return hex.EncodeToString(b[:])
}

var ownershipID = regexp.MustCompile(`^[a-f0-9]{32}$`)
var instanceID = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,128}$`)

func workerName(owner, attempt string) string { return "relay-" + owner[:12] + "-" + attempt }

// An empty directory is an explicitly in-memory journal for embedded/test callers.
// LoadConfigFile supplies a durable directory for the production binary.
func openJournal(directory string) (j *ownershipJournal, err error) {
	j = &ownershipJournal{directory: directory, data: journalData{Version: 1, OwnerID: uniqueID()}}
	if directory == "" {
		return j, nil
	}
	if err = os.MkdirAll(directory, 0700); err != nil {
		return nil, err
	}
	j.lock, err = os.OpenFile(filepath.Join(directory, "worker.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err = syscall.Flock(int(j.lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		j.lock.Close()
		return nil, fmt.Errorf("ownership journal already in use: %w", err)
	}
	lockedJournal := j
	defer func() {
		if err != nil {
			lockedJournal.close()
		}
	}()
	file, err := os.Open(filepath.Join(directory, "worker.json"))
	if errors.Is(err, os.ErrNotExist) {
		err = j.save(nil)
		return j, err
	}
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() > 64<<10 {
		return nil, errors.New("ownership journal exceeds size limit")
	}
	j.data = journalData{}
	decoder := json.NewDecoder(io.LimitReader(file, 64<<10))
	decoder.DisallowUnknownFields()
	if err = decoder.Decode(&j.data); err != nil {
		return nil, fmt.Errorf("invalid ownership journal: %w", err)
	}
	var extra any
	if err = decoder.Decode(&extra); err != io.EOF {
		return nil, errors.New("ownership journal contains trailing data")
	}
	if j.data.Version != 1 || !ownershipID.MatchString(j.data.OwnerID) {
		return nil, errors.New("invalid ownership identity or journal version")
	}
	if w := j.data.Worker; w != nil {
		if w.CreatedAt.IsZero() || !validWorkerPhase(w.Phase) || !ownershipID.MatchString(w.AttemptID) || w.Name != workerName(j.data.OwnerID, w.AttemptID) || (w.InstanceID != "" && !instanceID.MatchString(w.InstanceID)) {
			return nil, errors.New("invalid worker ownership record")
		}
	}
	return j, nil
}
func (j *ownershipJournal) save(worker *WorkerRecord) error {
	if j.closed {
		return errors.New("ownership journal is closed")
	}
	data := j.data
	data.Worker = nil
	if worker != nil {
		copy := *worker
		data.Worker = &copy
	}
	if j.directory == "" {
		j.data = data
		return nil
	}
	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	temporary, err := os.CreateTemp(j.directory, ".worker-*")
	if err != nil {
		return err
	}
	name := temporary.Name()
	defer os.Remove(name)
	if err = temporary.Chmod(0600); err == nil {
		_, err = temporary.Write(append(bytes, '\n'))
	}
	if err == nil {
		err = temporary.Sync()
	}
	err = errors.Join(err, temporary.Close())
	if err != nil {
		return err
	}
	if err = os.Rename(name, filepath.Join(j.directory, "worker.json")); err != nil {
		return err
	}
	directory, err := os.Open(j.directory)
	if err != nil {
		return err
	}
	err = errors.Join(directory.Sync(), directory.Close())
	if err != nil {
		return err
	}
	j.data = data
	return nil
}
func (j *ownershipJournal) close() error {
	if j == nil || j.closed {
		return nil
	}
	j.closed = true
	if j.lock != nil {
		return j.lock.Close()
	}
	return nil
}

func validWorkerPhase(phase string) bool {
	switch phase {
	case "creating", "uncertain", "loading", "ready", "stopping", "not_created":
		return true
	}
	return false
}
