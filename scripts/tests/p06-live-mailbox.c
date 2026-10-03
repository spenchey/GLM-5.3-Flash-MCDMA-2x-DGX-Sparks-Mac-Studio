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

enum { CTRL = 4096, PAYLOAD = 32768, MIB = 1 << 20 };

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) { perror("clock_gettime"); exit(2); }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void pause_ms(void) {
    const struct timespec pause = {0, 1000000};
    nanosleep(&pause, NULL);
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
        const uint8_t want = pattern(i, n, lane);
        if (p[i] != want) {
            fprintf(stderr, "%s mismatch offset=%zu got=%u expected=%u\n", what, i, p[i], want);
            exit(1);
        }
    }
}

static void dump(const char *path, const uint8_t *p, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) { perror(path); exit(2); }
    for (size_t off = 0; off < n;) {
        ssize_t wrote = write(fd, p + off, n - off);
        if (wrote <= 0) { perror("write"); exit(2); }
        off += (size_t)wrote;
    }
    if (close(fd)) { perror("close"); exit(2); }
}

static int unix_command(const char *path, const char *line) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); exit(2); }
    struct sockaddr_un address = {0};
    address.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof(address.sun_path)) { fprintf(stderr, "socket path too long\n"); exit(2); }
    strcpy(address.sun_path, path);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address))) { perror("connect"); exit(2); }
    char command[64];
    snprintf(command, sizeof(command), "%s\n", line);
    if (write(fd, command, strlen(command)) != (ssize_t)strlen(command)) { perror("write socket"); exit(2); }
    char reply[32] = {0};
    ssize_t got = read(fd, reply, sizeof(reply) - 1);
    if (got <= 0 || strncmp(reply, "OK\n", 3)) {
        fprintf(stderr, "daemon refused %s: %s\n", line, got > 0 ? reply : "EOF");
        exit(1);
    }
    return fd;
}

static uint8_t *map_box(const char *name, uint64_t *req, uint64_t *rep, size_t *total) {
    char shm[128];
    snprintf(shm, sizeof(shm), "/mcdma-rpc.%s", name);
    int fd = shm_open(shm, O_RDWR, 0);
    if (fd < 0) { perror("shm_open"); exit(2); }
    struct stat st;
    if (fstat(fd, &st)) { perror("fstat"); exit(2); }
    uint8_t *box = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (box == MAP_FAILED) { perror("mmap"); exit(2); }
    memcpy(req, box + 256, 8);
    memcpy(rep, box + 264, 8);
    *total = (size_t)st.st_size;
    if (*req < 4 * MIB || *rep < 4 * MIB || *req + *rep != *total) {
        fprintf(stderr, "invalid mailbox layout\n"); exit(2);
    }
    return box;
}

static unsigned timeout_arg(const char *value) {
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (!parsed || parsed > 120 || *end) { fprintf(stderr, "timeout must be 1..120\n"); exit(2); }
    return (unsigned)parsed;
}

static void wait_ready(uint8_t *box, uint64_t deadline) {
    while (now_ns() < deadline && load_word((volatile uint64_t *)(box + 64)) != 1) pause_ms();
    if (load_word((volatile uint64_t *)(box + 64)) != 1) { fprintf(stderr, "link not ready\n"); exit(1); }
}

static void client_one(const char *name, const char *request_path, const char *reply_path, unsigned seconds) {
    uint64_t req, rep; size_t total;
    uint8_t *box = map_box(name, &req, &rep, &total);
    uint64_t deadline = now_ns() + (uint64_t)seconds * 1000000000ull;
    wait_ready(box, deadline);
    uint64_t generation = load_word((volatile uint64_t *)(box + 72));
    fill(box + CTRL, PAYLOAD, 0x31);
    verify(box + CTRL, PAYLOAD, 0x31, "request");
    dump(request_path, box + CTRL, PAYLOAD);
    store_word((volatile uint64_t *)box, word(1, PAYLOAD));
    uint64_t done = 0;
    while (now_ns() < deadline) {
        done = load_word((volatile uint64_t *)(box + req + 64));
        if (word_seq(done) == 1) break;
        pause_ms();
    }
    if (word_seq(done) != 1 || word_len(done) != PAYLOAD) { fprintf(stderr, "reply timeout or length mismatch\n"); exit(1); }
    if (load_word((volatile uint64_t *)(box + 72)) != generation) { fprintf(stderr, "generation changed\n"); exit(1); }
    verify(box + req + CTRL, PAYLOAD, 0xa7, "reply");
    dump(reply_path, box + req + CTRL, PAYLOAD);
    printf("{\"generation\":%llu,\"partial_reply\":false}\n", (unsigned long long)generation);
    munmap(box, total);
}

static void service_one(const char *name, const char *sock, const char *request_path,
                        const char *reply_path, unsigned seconds) {
    int service = unix_command(sock, "MODE poll");
    uint64_t req, rep; size_t total;
    uint8_t *box = map_box(name, &req, &rep, &total);
    uint64_t deadline = now_ns() + (uint64_t)seconds * 1000000000ull;
    uint64_t staged = 0;
    while (now_ns() < deadline) {
        staged = load_word((volatile uint64_t *)box);
        if (word_seq(staged) == 1) break;
        pause_ms();
    }
    if (word_seq(staged) != 1 || word_len(staged) != PAYLOAD) { fprintf(stderr, "request timeout or length mismatch\n"); exit(1); }
    verify(box + CTRL, PAYLOAD, 0x31, "request");
    dump(request_path, box + CTRL, PAYLOAD);
    fill(box + req + CTRL, PAYLOAD, 0xa7);
    verify(box + req + CTRL, PAYLOAD, 0xa7, "reply");
    dump(reply_path, box + req + CTRL, PAYLOAD);
    store_word((volatile uint64_t *)(box + req + 128), word(1, PAYLOAD));
    /* Keep the service registration alive until the listener has consumed the
       staged word. Closing immediately races the listener's socket EOF check
       and can discard a valid reply before either RDMA reply mode sends it. */
    uint64_t accepted = 0;
    while (now_ns() < deadline) {
        accepted = load_word((volatile uint64_t *)(box + req));
        if (word_seq(accepted) == 1) break;
        pause_ms();
    }
    if (word_seq(accepted) != 1 || word_len(accepted) != PAYLOAD) {
        fprintf(stderr, "reply acknowledgement timeout or length mismatch\n");
        exit(1);
    }
    printf("{\"served\":true}\n");
    close(service);
    munmap(box, total);
}

static void hold_service(const char *sock, unsigned seconds) {
    int service = unix_command(sock, "MODE poll");
    int flags = fcntl(service, F_GETFL, 0);
    if (flags < 0 || fcntl(service, F_SETFL, flags | O_NONBLOCK)) { perror("fcntl"); exit(2); }
    uint64_t deadline = now_ns() + (uint64_t)seconds * 1000000000ull;
    char reply[64];
    while (now_ns() < deadline) {
        ssize_t got = recv(service, reply, sizeof(reply), 0);
        if (got == 0) { puts("{\"disconnected\":true,\"reason\":\"eof\"}"); close(service); return; }
        if (got > 0) { puts("{\"disconnected\":true,\"reason\":\"message\"}"); close(service); return; }
        if (errno != EAGAIN && errno != EWOULDBLOCK) { perror("recv"); exit(1); }
        pause_ms();
    }
    fprintf(stderr, "service disconnect timeout\n");
    close(service);
    exit(1);
}

static void stage(const char *name, const char *request_path) {
    uint64_t req, rep; size_t total;
    uint8_t *box = map_box(name, &req, &rep, &total);
    if (load_word((volatile uint64_t *)(box + 64)) != 1) { fprintf(stderr, "link not ready\n"); exit(1); }
    fill(box + CTRL, PAYLOAD, 0x31);
    verify(box + CTRL, PAYLOAD, 0x31, "request");
    dump(request_path, box + CTRL, PAYLOAD);
    store_word((volatile uint64_t *)box, word(1, PAYLOAD));
    printf("{\"generation\":%llu}\n", (unsigned long long)load_word((volatile uint64_t *)(box + 72)));
    munmap(box, total);
}

static void wait_reply(const char *name, unsigned seconds) {
    uint64_t req, rep; size_t total;
    uint8_t *box = map_box(name, &req, &rep, &total);
    uint64_t deadline = now_ns() + (uint64_t)seconds * 1000000000ull;
    uint64_t done = 0;
    while (now_ns() < deadline) {
        done = load_word((volatile uint64_t *)(box + req + 64));
        if (word_seq(done) == 1) break;
        pause_ms();
    }
    if (word_seq(done) == 1) printf("{\"outcome\":\"complete\",\"reply_present\":true}\n");
    else printf("{\"outcome\":\"timeout\",\"reply_present\":false}\n");
    munmap(box, total);
}

static void inspect(const char *name) {
    uint64_t req, rep; size_t total;
    uint8_t *box = map_box(name, &req, &rep, &total);
    uint64_t staged = load_word((volatile uint64_t *)box);
    uint64_t done = load_word((volatile uint64_t *)(box + req + 64));
    printf("{\"ready\":%s,\"generation\":%llu,\"request_seq\":%u,\"reply_seq\":%u}\n",
           load_word((volatile uint64_t *)(box + 64)) == 1 ? "true" : "false",
           (unsigned long long)load_word((volatile uint64_t *)(box + 72)),
           word_seq(staged), word_seq(done));
    munmap(box, total);
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "inspect")) inspect(argv[2]);
    else if (argc == 4 && !strcmp(argv[1], "stage")) stage(argv[2], argv[3]);
    else if (argc == 4 && !strcmp(argv[1], "wait")) wait_reply(argv[2], timeout_arg(argv[3]));
    else if (argc == 6 && !strcmp(argv[1], "client-one"))
        client_one(argv[2], argv[3], argv[4], timeout_arg(argv[5]));
    else if (argc == 7 && !strcmp(argv[1], "service-one"))
        service_one(argv[2], argv[3], argv[4], argv[5], timeout_arg(argv[6]));
    else if (argc == 4 && !strcmp(argv[1], "hold-service"))
        hold_service(argv[2], timeout_arg(argv[3]));
    else {
        fprintf(stderr, "usage: %s inspect NAME | stage NAME REQUEST | wait NAME SECONDS | client-one NAME REQUEST REPLY SECONDS | service-one NAME SOCKET REQUEST REPLY SECONDS | hold-service SOCKET SECONDS\n", argv[0]);
        return 2;
    }
    return 0;
}
