// tfidf_serial.c  —  Serial TF-IDF in C (C17, MSVC-friendly, no external headers)
// Build: Visual Studio → Console App (C), x64, C17 (/std:c17)
// Run:   YourExe.exe doclist.txt out_dir
#define _CRT_SECURE_NO_WARNINGS
#ifdef _MSC_VER
#define strdup _strdup
#endif

#include <ctype.h>
#include <direct.h>   // _mkdir (Windows)
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------- Minimal string->int hashmap (open addressing, no deletes) ----------
typedef struct { char* key; int value; int used; } HMEntry;
typedef struct { HMEntry* a; size_t cap; size_t size; } HashMapI;

static uint64_t fnv1a64(const char* s){
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char* p=(const unsigned char*)s; *p; ++p){
        h ^= (uint64_t)*p;
        h *= 1099511628211ULL;
    }
    return h;
}
static size_t next_pow2(size_t x){ size_t p=1; while(p<x) p<<=1; return p; }

static void hm_init(HashMapI* m, size_t cap){
    if (cap < 16) cap = 16;
    cap = next_pow2(cap);
    m->a = (HMEntry*)calloc(cap, sizeof(HMEntry));
    m->cap = cap; m->size = 0;
    if(!m->a){ fprintf(stderr,"alloc fail\n"); exit(1); }
}
static void hm_free(HashMapI* m){
    if(!m || !m->a) return;
    for (size_t i=0;i<m->cap;++i) if (m->a[i].used) free(m->a[i].key);
    free(m->a); m->a=NULL; m->cap=0; m->size=0;
}
static void hm_rehash(HashMapI* m, size_t newcap){
    HashMapI n; hm_init(&n, newcap);
    for(size_t i=0;i<m->cap;++i){
        if(!m->a[i].used) continue;
        char* k = m->a[i].key; int v = m->a[i].value;
        uint64_t h = fnv1a64(k); size_t mask = n.cap-1; size_t j = (size_t)(h & mask);
        while(n.a[j].used) j = (j+1) & mask;
        n.a[j].used = 1; n.a[j].key = k; n.a[j].value = v; n.size++;
    }
    free(m->a);
    *m = n; // move
}
static void hm_maybe_grow(HashMapI* m){
    // grow when load factor > 0.7
    if ((m->size+1)*10 >= (m->cap*7)) hm_rehash(m, m->cap<<1);
}
// increment key by delta (insert if missing)
static void hm_add(HashMapI* m, const char* key, int delta){
    hm_maybe_grow(m);
    uint64_t h = fnv1a64(key); size_t mask = m->cap - 1; size_t i = (size_t)(h & mask);
    while (m->a[i].used){
        if (strcmp(m->a[i].key, key)==0){ m->a[i].value += delta; return; }
        i = (i+1) & mask;
    }
    m->a[i].used = 1;
    m->a[i].key = strdup(key);
    m->a[i].value = delta;
    m->size++;
}
// get value (0 if missing)
static int hm_get(const HashMapI* m, const char* key){
    if(m->size==0) return 0;
    uint64_t h = fnv1a64(key); size_t mask = m->cap - 1; size_t i = (size_t)(h & mask);
    while (m->a[i].used){
        if (strcmp(m->a[i].key, key)==0) return m->a[i].value;
        i = (i+1) & mask;
    }
    return 0;
}
// ---------- End hashmap ----------

// read lines from a file into array of strings
static char** read_lines(const char* path, int* outN){
    FILE* f = fopen(path, "rb");
    if(!f){ fprintf(stderr,"Impossibile aprire %s\n", path); return NULL; }
    size_t cap=128, n=0; char** lines=(char**)malloc(cap*sizeof(char*));
    char buf[8192];
    while (fgets(buf, sizeof buf, f)){
        size_t L=strlen(buf);
        while (L && (buf[L-1]=='\n' || buf[L-1]=='\r')) buf[--L]=0;
        if(!L) continue;
        if(n==cap){ cap*=2; lines=(char**)realloc(lines, cap*sizeof(char*)); }
        lines[n++] = strdup(buf);
    }
    fclose(f);
    *outN = (int)n; return lines;
}

// read whole file into memory
static char* read_file(const char* path, size_t* outLen){
    FILE* f = fopen(path, "rb");
    if(!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if(sz < 0){ fclose(f); return NULL; }
    char* data = (char*)malloc((size_t)sz + 1);
    if(!data){ fclose(f); return NULL; }
    size_t got = fread(data, 1, (size_t)sz, f);
    fclose(f);
    data[got]=0; if(outLen) *outLen = got; return data;
}

// tokenize: keep letters, lowercase; split on non-letters
static void process_document(const char* text, HashMapI* tf, HashMapI* seen){
    hm_init(tf, 128);
    hm_init(seen, 128);
    size_t cap=64, len=0; char* tok=(char*)malloc(cap);
    for (const unsigned char* p=(const unsigned char*)text;; ++p){
        int c = *p;
        int alpha = c && isalpha(c);
        if (alpha){
            char ch = (char)tolower(c);
            if(len+1>=cap){ cap*=2; tok=(char*)realloc(tok, cap); }
            tok[len++] = ch;
        }
        if (!alpha){
            if (len){
                tok[len]=0;
                hm_add(tf, tok, 1);
                // set semantics: only mark once
                if (hm_get(seen, tok)==0) hm_add(seen, tok, 1);
                len=0;
            }
            if (!c) break;
        }
    }
    free(tok);
}

typedef struct { char* term; double w; } PairSD;
static int cmp_desc(const void* a, const void* b){
    double da = ((const PairSD*)a)->w;
    double db = ((const PairSD*)b)->w;
    return (db>da) - (db<da);
}

int main(int argc, char** argv){
    setlocale(LC_ALL, ""); // per lettere accentate nei testi locali

    if (argc < 3){
        fprintf(stderr, "Uso: %s doclist.txt out_dir\n", argv[0]);
        return 1;
    }
    const char* doclist = argv[1];
    const char* outdir  = argv[2];

    int N=0; char** docs = read_lines(doclist, &N);
    if(!docs || N==0){ fprintf(stderr,"doclist vuota o non leggibile.\n"); return 1; }

    _mkdir(outdir); // crea dir se non esiste

    // Per ogni documento: TF e set "seen"; accumula DF globale.
    HashMapI df; hm_init(&df, 1<<10); // document frequency globale
    HashMapI* doc_tf = (HashMapI*)calloc(N, sizeof(HashMapI));
    if (!doc_tf){ fprintf(stderr,"alloc doc_tf fail\n"); return 1; }

    for (int i=0;i<N;++i){
        size_t L=0; char* txt = read_file(docs[i], &L);
        if(!txt){ fprintf(stderr,"Impossibile leggere: %s\n", docs[i]); hm_init(&doc_tf[i],16); continue; }

        HashMapI tf, seen;
        process_document(txt, &tf, &seen);
        free(txt);
        doc_tf[i] = tf;

        // aggiorna DF con i termini unici del documento
        for (size_t s=0; s<seen.cap; ++s){
            if(!seen.a[s].used) continue;
            hm_add(&df, seen.a[s].key, 1);
        }
        hm_free(&seen);
    }

    // Scrivi TF-IDF top-k per documento
    int k = 20;
    char outpath[1024]; snprintf(outpath, sizeof outpath, "%s/%s", outdir, "tfidf_serial.txt");
    FILE* out = fopen(outpath, "wb");
    if(!out){ fprintf(stderr,"Impossibile aprire output '%s'\n", outpath); return 1; }

    for (int i=0;i<N;++i){
        // conta termini nel documento
        size_t cnt = doc_tf[i].size;
        fprintf(out, "DOC %d %s\n", i, docs[i]);
        if (cnt == 0){ fprintf(out, "\n"); continue; }

        PairSD* arr = (PairSD*)malloc(cnt * sizeof(PairSD));
        size_t idx=0;
        for (size_t s=0; s<doc_tf[i].cap; ++s){
            if(!doc_tf[i].a[s].used) continue;
            char* term = doc_tf[i].a[s].key;
            int tf = doc_tf[i].a[s].value;
            int df_count = hm_get(&df, term);
            double idf = log((N + 1.0) / (df_count + 1.0)) + 1.0; // smooth idf
            arr[idx].term = term;
            arr[idx].w = tf * idf;
            idx++;
        }
        // ordina desc e stampa top-k
        qsort(arr, idx, sizeof(PairSD), cmp_desc);
        int limit = (k < (int)idx ? k : (int)idx);
        for (int j=0;j<limit;++j) fprintf(out, "%s\t%.6f\n", arr[j].term, arr[j].w);
        fprintf(out, "\n");
        free(arr);
    }
    fclose(out);

    // cleanup
    for (int i=0;i<N;++i){ hm_free(&doc_tf[i]); free(docs[i]); }
    free(doc_tf); free(docs);
    hm_free(&df);

    printf("OK. Output: %s\n", outpath);
    return 0;
}
 