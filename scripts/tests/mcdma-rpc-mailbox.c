#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

enum { CTRL = 4096, MIB = 1 << 20 };
static const size_t sizes[] = {32768, 2 * MIB};

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) { perror("clock_gettime"); exit(2); }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t load_word(volatile uint64_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static void store_word(volatile uint64_t *p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static uint64_t word(uint32_t seq, uint32_t len) { return ((uint64_t)seq << 32) | len; }
static uint32_t word_seq(uint64_t v) { return (uint32_t)(v >> 32); }
static uint32_t word_len(uint64_t v) { return (uint32_t)v; }

static uint8_t pattern(size_t i, size_t n, uint8_t lane) {
    uint64_t x = (uint64_t)i * 0x9e3779b185ebca87ull;
    x ^= (uint64_t)n * 0xd6e8feb86659fd93ull;
    x ^= (uint64_t)lane * 0xa0761d6478bd642full;
    x ^= x >> 29;
    x *= 0xbf58476d1ce4e5b9ull;
    return (uint8_t)(x ^ (x >> 17) ^ (x >> 41));
}

static void fill(uint8_t *p, size_t n, uint8_t lane) {
    for (size_t i = 0; i < n; ++i) p[i] = pattern(i, n, lane);
}

static void verify(const uint8_t *p, size_t n, uint8_t lane, const char *what) {
    for (size_t i = 0; i < n; ++i) {
        uint8_t want = pattern(i, n, lane);
        if (p[i] != want) {
            fprintf(stderr, "%s mismatch offset=%zu got=%u expected=%u\n", what, i, p[i], want);
            exit(1);
        }
    }
}

static void dump(const char *dir, const char *side, const char *kind, size_t n, const uint8_t *p) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s-%s-%zu.bin", dir, side, kind, n);
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) { perror(path); exit(2); }
    size_t off = 0;
    while (off < n) {
        ssize_t wrote = write(fd, p + off, n - off);
        if (wrote <= 0) { perror("write"); exit(2); }
        off += (size_t)wrote;
    }
    if (close(fd)) { perror("close"); exit(2); }
}

static uint64_t wait_seq(volatile uint64_t *p, uint32_t seq, int equal, uint64_t deadline) {
    while (now_ns() < deadline) {
        uint64_t v = load_word(p);
        uint32_t got = word_seq(v);
        if ((equal && got == seq) || (!equal && got && got != seq)) return v;
        struct timespec pause = {0, 1000000};
        nanosleep(&pause, NULL);
    }
    return 0;
}

static void pause_one_ms(void) {
    struct timespec pause = {0, 1000000};
    nanosleep(&pause, NULL);
}

static int unix_command(const char *path, const char *line, int keep_open) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); exit(2); }
    struct sockaddr_un a = {0};
    a.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof(a.sun_path)) { fprintf(stderr, "socket path too long\n"); exit(2); }
    strcpy(a.sun_path, path);
    if (connect(fd, (struct sockaddr *)&a, sizeof(a))) { perror("connect socket"); exit(2); }
    char out[64];
    snprintf(out, sizeof(out), "%s\n", line);
    if (write(fd, out, strlen(out)) != (ssize_t)strlen(out)) { perror("write socket"); exit(2); }
    char reply[16] = {0};
    ssize_t got = read(fd, reply, sizeof(reply) - 1);
    if (got <= 0) { perror("read socket"); exit(2); }
    if (strncmp(reply, "OK\n", 3)) { fprintf(stderr, "daemon refused %s: %s", line, reply); exit(1); }
    if (!keep_open) close(fd);
    return fd;
}

static uint8_t *map_box(const char *name, uint64_t *req, uint64_t *rep, size_t *total) {
    char shm[64];
    snprintf(shm, sizeof(shm), "/mcdma-rpc.%s", name);
    int fd = shm_open(shm, O_RDWR, 0);
    if (fd < 0) { perror("shm_open"); exit(2); }
    struct stat st;
    if (fstat(fd, &st)) { perror("fstat"); exit(2); }
    uint8_t *p = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); exit(2); }
    memcpy(req, p + 256, 8);
    memcpy(rep, p + 264, 8);
    *total = (size_t)st.st_size;
    if (!*req || !*rep || *req + *rep != *total || *req < 4 * MIB || *rep < 4 * MIB) {
        fprintf(stderr, "invalid mailbox layout req=%llu rep=%llu total=%zu\n",
                (unsigned long long)*req, (unsigned long long)*rep, *total);
        exit(2);
    }
    return p;
}

static void run_client(const char *name, const char *dir, unsigned timeout_s) {
    uint64_t req, rep; size_t total;
    uint8_t *box = map_box(name, &req, &rep, &total);
    uint64_t deadline = now_ns() + (uint64_t)timeout_s * 1000000000ull;
    while (now_ns() < deadline && load_word((volatile uint64_t *)(box + 64)) != 1) pause_one_ms();
    uint64_t generation = load_word((volatile uint64_t *)(box + 72));
    if (load_word((volatile uint64_t *)(box + 64)) != 1 || !generation) {
        fprintf(stderr, "link did not become ready\n"); exit(1);
    }
    for (uint32_t seq = 1; seq <= 2; ++seq) {
        size_t n = sizes[seq - 1];
        fill(box + CTRL, n, 0x31);
        verify(box + CTRL, n, 0x31, "client request");
        dump(dir, "client", "request", n, box + CTRL);
        store_word((volatile uint64_t *)box, word(seq, (uint32_t)n));
        uint64_t done = wait_seq((volatile uint64_t *)(box + req + 64), seq, 1, deadline);
        if (!done || word_len(done) != n) { fprintf(stderr, "reply timeout/length seq=%u word=%llu\n", seq, (unsigned long long)done); exit(1); }
        if (load_word((volatile uint64_t *)(box + 72)) != generation) { fprintf(stderr, "generation changed during call\n"); exit(1); }
        verify(box + req + CTRL, n, 0xa7, "client reply");
        dump(dir, "client", "reply", n, box + req + CTRL);
        printf("CLIENT_PASS size=%zu seq=%u generation=%llu\n", n, seq, (unsigned long long)generation);
    }
    munmap(box, total);
}

static void run_service(const char *name, const char *sock, const char *dir, unsigned timeout_s) {
    int service = unix_command(sock, "MODE poll", 1);
    uint64_t req, rep; size_t total;
    uint8_t *box = map_box(name, &req, &rep, &total);
    uint64_t deadline = now_ns() + (uint64_t)timeout_s * 1000000000ull;
    uint32_t last = 0;
    for (uint32_t call = 1; call <= 2; ++call) {
        uint64_t staged = wait_seq((volatile uint64_t *)box, last, 0, deadline);
        uint32_t seq = word_seq(staged), n = word_len(staged);
        if (!staged || n != sizes[call - 1]) { fprintf(stderr, "request timeout/length call=%u word=%llu\n", call, (unsigned long long)staged); exit(1); }
        verify(box + CTRL, n, 0x31, "service request");
        dump(dir, "service", "request", n, box + CTRL);
        fill(box + req + CTRL, n, 0xa7);
        verify(box + req + CTRL, n, 0xa7, "service reply");
        dump(dir, "service", "reply", n, box + req + CTRL);
        store_word((volatile uint64_t *)(box + req + 128), word(seq, n));
        printf("SERVICE_PASS size=%u seq=%u\n", n, seq);
        last = seq;
    }
    close(service);
    munmap(box, total);
}

int main(int argc, char **argv) {
    if (argc != 6 || (strcmp(argv[1], "client") && strcmp(argv[1], "service"))) {
        fprintf(stderr, "usage: %s {client|service} NAME SOCKET DUMP_DIR TIMEOUT_SECONDS\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    unsigned long timeout = strtoul(argv[5], &end, 10);
    if (!timeout || timeout > 120 || *end) { fprintf(stderr, "timeout must be 1..120 seconds\n"); return 2; }
    if (!strcmp(argv[1], "client")) run_client(argv[2], argv[4], (unsigned)timeout);
    else run_service(argv[2], argv[3], argv[4], (unsigned)timeout);
    return 0;
}
