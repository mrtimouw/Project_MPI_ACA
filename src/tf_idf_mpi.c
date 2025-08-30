// tfidf_mpi.c — Parallel TF-IDF with MPI (C17, MSVC-friendly, UTF-8 robust via WinAPI, no external headers)
// Build (Dev Prompt): cl /TC /std:c17 /O2 tfidf_mpi.c /Fe:tfidf_mpi.exe /I"%ProgramFiles(x86)%\Microsoft SDKs\MPI\Include" /link /LIBPATH:"%ProgramFiles(x86)%\Microsoft SDKs\MPI\Lib\x64" msmpi.lib
// Run: mpiexec -n 8 tfidf_mpi.exe doclist.txt out_dir
#define _CRT_SECURE_NO_WARNINGS
#ifdef _MSC_VER
#define strdup _strdup
#endif

#include <mpi.h>
#include <windows.h>
#include <ctype.h>
#include <direct.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>

// ===== Minimal string->int hashmap (open addressing) =====
typedef struct { char* key; int value; int used; } HMEntry;
typedef struct { HMEntry* a; size_t cap; size_t size; } HashMapI;

static uint64_t fnv1a64(const char* s){
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char* p=(const unsigned char*)s; *p; ++p){ h ^= (uint64_t)*p; h *= 1099511628211ULL; }
    return h;
}
static size_t next_pow2(size_t x){ size_t p=1; while(p<x) p<<=1; return p; }
static void hm_init(HashMapI* m, size_t cap){
    if (cap < 16) cap = 16; cap = next_pow2(cap);
    m->a = (HMEntry*)calloc(cap, sizeof(HMEntry));
    if(!m->a){ fprintf(stderr,"alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    m->cap = cap; m->size = 0;
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
    free(m->a); *m = n;
}
static void hm_maybe_grow(HashMapI* m){ if ((m->size+1)*10 >= (m->cap*7)) hm_rehash(m, m->cap<<1); }
static void hm_add(HashMapI* m, const char* key, int delta){
    hm_maybe_grow(m);
    uint64_t h = fnv1a64(key); size_t mask = m->cap - 1; size_t i = (size_t)(h & mask);
    while (m->a[i].used){ if (strcmp(m->a[i].key, key)==0){ m->a[i].value += delta; return; } i = (i+1) & mask; }
    m->a[i].used = 1; m->a[i].key = strdup(key); m->a[i].value = delta; m->size++;
}
static int hm_get(const HashMapI* m, const char* key){
    if(!m || m->size==0) return 0;
    uint64_t h = fnv1a64(key); size_t mask = m->cap - 1; size_t i = (size_t)(h & mask);
    while (m->a[i].used){ if (strcmp(m->a[i].key, key)==0) return m->a[i].value; i = (i+1) & mask; }
    return 0;
}
// ===== End hashmap =====

// ===== UTF-8 conversions via WinAPI (independent from locale) =====
static wchar_t* utf8_to_wide(const char* s){
    if(!s) return NULL;
    int lenW = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    if (lenW <= 0) return NULL;
    wchar_t* w = (wchar_t*)malloc(lenW * sizeof(wchar_t));
    if(!w) return NULL;
    if (!MultiByteToWideChar(CP_UTF8, 0, s, -1, w, lenW)) { free(w); return NULL; }
    if (w[0] == 0xFEFF) { memmove(w, w+1, (lenW - 1) * sizeof(wchar_t)); }
    return w;
}
static char* wide_to_utf8(const wchar_t* w){
    if(!w) return NULL;
    int lenU8 = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if (lenU8 <= 0) return NULL;
    char* s = (char*)malloc(lenU8);
    if(!s) return NULL;
    if (!WideCharToMultiByte(CP_UTF8, 0, w, -1, s, lenU8, NULL, NULL)) { free(s); return NULL; }
    return s;
}

// ===== I/O utils =====
static char** read_lines(const char* path, int* outN){
    FILE* f = fopen(path, "rb");
    if(!f){ fprintf(stderr,"Impossibile aprire %s\n", path); return NULL; }
    size_t cap=128, n=0; char** lines=(char**)malloc(cap*sizeof(char*));
    char buf[8192];
    while (fgets(buf, sizeof buf, f)){
        size_t L=strlen(buf);
        while (L && (buf[L-1]=='\n' || buf[L-1]=='\r')) buf[--L]=0;
        if (n==0 && L>=3 && (unsigned char)buf[0]==0xEF && (unsigned char)buf[1]==0xBB && (unsigned char)buf[2]==0xBF){
            memmove(buf, buf+3, L-2); L-=3; buf[L]=0;
        }
        if(!L) continue;
        if(n==cap){ cap*=2; lines=(char**)realloc(lines, cap*sizeof(char*)); }
        lines[n++] = strdup(buf);
    }
    fclose(f);
    *outN = (int)n; return lines;
}
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

// ===== Tokenization (UTF-8 aware) =====
static void process_document(const char* text, HashMapI* tf, HashMapI* seen){
    hm_init(tf, 128);
    hm_init(seen, 128);

    wchar_t* wbuf = utf8_to_wide(text);
    if(!wbuf){
        size_t cap=64, len=0; char* tok=(char*)malloc(cap);
        for (const unsigned char* p=(const unsigned char*)text;; ++p){
            int c=*p; int alpha = c && isalpha(c);
            if(alpha){ char ch=(char)tolower(c); if(len+1>=cap){ cap*=2; tok=(char*)realloc(tok,cap);} tok[len++]=ch; }
            if(!alpha){ if(len){ tok[len]=0; hm_add(tf,tok,1); if(hm_get(seen,tok)==0) hm_add(seen,tok,1); len=0; } if(!c) break; }
        }
        free(tok);
        return;
    }

    size_t outLen = wcslen(wbuf);
    size_t capW=64, tlen=0; wchar_t* wtok=(wchar_t*)malloc(capW*sizeof(wchar_t));
    for (size_t i=0; i<=outLen; ++i){
        wchar_t wc = wbuf[i];
        int alpha = (wc != 0 && iswalpha(wc));
        if (alpha){
            if (tlen+1>=capW){ capW*=2; wtok=(wchar_t*)realloc(wtok, capW*sizeof(wchar_t)); }
            wtok[tlen++] = towlower(wc);
        }
        if (!alpha){
            if (tlen){
                wtok[tlen]=0;
                char* tok8 = wide_to_utf8(wtok);
                if (tok8){
                    // filtro semplice di lunghezza, utile per ridurre rumore/memoria
                    size_t L = strlen(tok8);
                    if (L >= 2 && L <= 40){
                        hm_add(tf, tok8, 1);
                        if (hm_get(seen, tok8)==0) hm_add(seen, tok8, 1);
                    }
                    free(tok8);
                }
                tlen=0;
            }
            if (!wc) break;
        }
    }
    free(wtok); free(wbuf);
}

// ===== Serialization helpers for DF/dictionary =====
static char* serialize_df(const HashMapI* df, int* outLen){
    // "term\tcount\n"
    size_t est = df->size * 16 + 1;
    char* s = (char*)malloc(est); size_t pos=0;
    if(!s){ *outLen=0; return NULL; }
    for(size_t i=0;i<df->cap;++i){
        if(!df->a[i].used) continue;
        const char* k = df->a[i].key; int v = df->a[i].value;
        int need = (int)strlen(k) + 1 + 12 + 1;
        if (pos + need + 1 > est){ est = (est*3)/2 + need + 64; s = (char*)realloc(s, est); }
        pos += sprintf(s+pos, "%s\t%d\n", k, v);
    }
    s[pos]=0; *outLen=(int)pos; return s;
}
static void deserialize_df_and_merge(const char* buf, int len, HashMapI* df_merge){
    const char* p = buf; const char* end = buf + len;
    while (p < end){
        const char* tab = (const char*)memchr(p, '\t', end-p);
        if(!tab) break;
        const char* nl = (const char*)memchr(tab+1, '\n', end-(tab+1));
        if(!nl) nl = end;
        int klen = (int)(tab - p);
        int v = atoi(tab+1);
        char* key = (char*)malloc(klen+1);
        memcpy(key, p, klen); key[klen]=0;
        hm_add(df_merge, key, v);
        free(key);
        p = nl + (nl<end);
    }
}
typedef struct { char* term; double idf; } DictEntry;
typedef struct { DictEntry* a; size_t n, cap; } DictList;

static void dictlist_init(DictList* L){ L->a=NULL; L->n=0; L->cap=0; }
static void dictlist_push(DictList* L, const char* term, double idf){
    if (L->n == L->cap){ L->cap = L->cap? L->cap*2 : 1024; L->a = (DictEntry*)realloc(L->a, L->cap*sizeof(DictEntry)); }
    L->a[L->n].term = strdup(term); L->a[L->n].idf = idf; L->n++;
}
static void dictlist_free(DictList* L){
    for (size_t i=0;i<L->n;++i) free(L->a[i].term);
    free(L->a); L->a=NULL; L->n=L->cap=0;
}
static char* serialize_dict(const DictList* L, int* outLen){
    // "term\tidf\n"
    size_t est = L->n * 24 + 1;
    char* s = (char*)malloc(est); size_t pos=0;
    for (size_t i=0;i<L->n;++i){
        const char* k = L->a[i].term; double idf=L->a[i].idf;
        int need = (int)strlen(k) + 1 + 32 + 1;
        if (pos + need + 1 > est){ est = (est*3)/2 + need + 64; s = (char*)realloc(s, est); }
        pos += sprintf(s+pos, "%s\t%.12g\n", k, idf);
    }
    s[pos]=0; *outLen=(int)pos; return s;
}
static void deserialize_dict_to_map(const char* buf, int len, HashMapI* idf_map_scaled){
    // memorizziamo idf*1e6 come int per efficienza (poi dividiamo)
    hm_init(idf_map_scaled, 1<<14);
    const char* p = buf; const char* end = buf + len;
    while (p < end){
        const char* tab = (const char*)memchr(p, '\t', end-p);
        if(!tab) break;
        const char* nl = (const char*)memchr(tab+1, '\n', end-(tab+1));
        if(!nl) nl = end;
        int klen = (int)(tab - p);
        double idf = atof(tab+1);
        char* key = (char*)malloc(klen+1);
        memcpy(key, p, klen); key[klen]=0;
        int scaled = (int)lround(idf * 1000000.0);
        hm_add(idf_map_scaled, key, scaled);
        free(key);
        p = nl + (nl<end);
    }
}

// ===== Main MPI program =====
typedef struct { char* term; double w; } PairSD;
static int cmp_desc(const void* a, const void* b){
    double da = ((const PairSD*)a)->w, db = ((const PairSD*)b)->w;
    return (db>da) - (db<da);
}

int main(int argc, char** argv){
    setlocale(LC_ALL, ".UTF-8");

    MPI_Init(&argc, &argv);
    int rank=0, P=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&P);

    if (argc < 3){
        if(rank==0) fprintf(stderr, "Uso: mpiexec -n P tfidf_mpi.exe doclist.txt out_dir\n");
        MPI_Finalize(); return 1;
    }
    const char* doclist_path = argv[1];
    const char* outdir = argv[2];

    // Rank 0 legge la doclist e broadcasta i percorsi
    int N = 0;
    char** docs = NULL;
    if (rank == 0){
        docs = read_lines(doclist_path, &N);
        if(!docs || N==0){ fprintf(stderr,"doclist vuota o non leggibile.\n"); }
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N <= 0){ MPI_Finalize(); return 1; }
    if (rank != 0) docs = (char**)malloc(N * sizeof(char*));

    // Broadcast lunghezze + contenuti stringhe (semplice ma robusto)
    int* lengths = (int*)malloc(N * sizeof(int));
    if (rank==0){ for (int i=0;i<N;++i) lengths[i] = (int)strlen(docs[i]); }
    MPI_Bcast(lengths, N, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i=0;i<N;++i){
        if (rank != 0) docs[i] = (char*)malloc(lengths[i]+1);
        MPI_Bcast(docs[i], lengths[i], MPI_CHAR, 0, MPI_COMM_WORLD);
        docs[i][lengths[i]] = 0;
    }

    // Ogni rank elabora la sua quota: TF per documento + DF locale
    HashMapI df_local; hm_init(&df_local, 1<<14);
    HashMapI* doc_tf = (HashMapI*)calloc(N, sizeof(HashMapI)); // solo per i doc del rank verrà usato
    for (int i=0;i<N;++i){
        if (i % P != rank) continue;
        size_t L=0; char* txt = read_file(docs[i], &L);
        if(!txt){ fprintf(stderr,"[rank %d] Impossibile leggere: %s\n", rank, docs[i]); hm_init(&doc_tf[i],16); continue; }
        HashMapI tf, seen; process_document(txt, &tf, &seen); free(txt);
        doc_tf[i] = tf;
        // aggiorna DF locale con i termini unici
        for (size_t s=0; s<seen.cap; ++s){ if(!seen.a[s].used) continue; hm_add(&df_local, seen.a[s].key, 1); }
        hm_free(&seen);
    }

    // Serializza DF locale e gather su rank 0
    int payload_len=0; char* payload = serialize_df(&df_local, &payload_len);
    int* sizes = NULL; int* displs = NULL;
    if (rank == 0){ sizes = (int*)malloc(P*sizeof(int)); }
    MPI_Gather(&payload_len, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char* recvbuf = NULL;
    if (rank == 0){
        displs = (int*)malloc(P*sizeof(int));
        int total=0; for (int r=0;r<P;++r){ displs[r]=total; total+=sizes[r]; }
        recvbuf = (char*)malloc(total>0? total : 1);
    }
    MPI_Gatherv(payload, payload_len, MPI_CHAR,
                recvbuf, sizes, displs, MPI_CHAR,
                0, MPI_COMM_WORLD);

    // Rank 0: merge DF globali, calcolo IDF, serializzazione dizionario
    HashMapI idf_map_scaled; // idf*1e6
    int dict_len=0; char* dict_ser=NULL;
    if (rank == 0){
        HashMapI df_global; hm_init(&df_global, 1<<16);
        if (recvbuf){
            for (int r=0; r<P; ++r){
                if (sizes[r] <= 0) continue;
                deserialize_df_and_merge(recvbuf + displs[r], sizes[r], &df_global);
            }
        }
        // Calcolo IDF e costruzione lista dizionario
        DictList D; dictlist_init(&D);
        for (size_t s=0; s<df_global.cap; ++s){
            if(!df_global.a[s].used) continue;
            const char* term = df_global.a[s].key; int dfc = df_global.a[s].value;
            double idf = log((N + 1.0) / (dfc + 1.0)) + 1.0;
            dictlist_push(&D, term, idf);
        }
        dict_ser = serialize_dict(&D, &dict_len);
        dictlist_free(&D);
        hm_free(&df_global);
        if (sizes) free(sizes);
        if (displs) free(displs);
        if (recvbuf) free(recvbuf);
    }
    if (payload) free(payload);
    hm_free(&df_local);

    // Broadcast del dizionario serializzato
    MPI_Bcast(&dict_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) dict_ser = (char*)malloc(dict_len>0? dict_len : 1);
    if (dict_len>0) MPI_Bcast(dict_ser, dict_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Deserializza dizionario in mappa: term -> (idf*1e6)
    deserialize_dict_to_map(dict_ser, dict_len, &idf_map_scaled);
    if (dict_ser) free(dict_ser);

    // Output TF-IDF locale (top-k)
    int k = 20;
    _mkdir(outdir);
    char partpath[1024]; sprintf(partpath, "%s/part_%d.txt", outdir, rank);
    FILE* out = fopen(partpath, "wb");
    if (!out){ fprintf(stderr,"[rank %d] Impossibile aprire output '%s'\n", rank, partpath); }

    for (int i=0;i<N;++i){
        if (i % P != rank) continue;
        size_t cnt = doc_tf[i].size;
        if (out) fprintf(out, "DOC %d %s\n", i, docs[i]);
        if (cnt == 0){ if(out) fprintf(out, "\n"); hm_free(&doc_tf[i]); continue; }

        PairSD* arr = (PairSD*)malloc(cnt * sizeof(PairSD)); size_t idx=0;
        for (size_t s=0; s<doc_tf[i].cap; ++s){
            if(!doc_tf[i].a[s].used) continue;
            char* term = doc_tf[i].a[s].key; int tf = doc_tf[i].a[s].value;
            int idf_scaled = hm_get(&idf_map_scaled, term);
            if (idf_scaled == 0) continue; // termine non presente? (dovrebbe essere raro)
            double idf = idf_scaled / 1000000.0;
            arr[idx].term = term; arr[idx].w = tf * idf; idx++;
        }
        qsort(arr, idx, sizeof(PairSD), cmp_desc);
        int limit = (k < (int)idx ? k : (int)idx);
        for (int j=0;j<limit;++j) if(out) fprintf(out, "%s\t%.6f\n", arr[j].term, arr[j].w);
        if (out) fprintf(out, "\n");
        free(arr);
        hm_free(&doc_tf[i]);
    }
    if (out) fclose(out);

    // cleanup docs + idf_map
    for (int i=0;i<N;++i) free(docs[i]); free(docs);
    hm_free(&idf_map_scaled);

    if (rank==0) printf("OK MPI. Output per-rank in: %s/part_*.txt\n", outdir);
    MPI_Finalize();
    return 0;
}
