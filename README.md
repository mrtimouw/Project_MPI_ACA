# Parallel TF–IDF with MPI

This project implements the calculation of **Term Frequency–Inverse Document Frequency (TF–IDF)** for a large corpus of text documents.  
The code has been developed both in a **serial version** (C) and in a **parallel version** using **MPI**.  
It was made as part of the *Advanced Computer Architecture* course at the University of Pavia.

------------------------------------------------------------
How to compile and run the program
------------------------------------------------------------

### Parallel version
cd ~/ACA-MPI-TFIDF/mpi/
mpicc -o tfidf_mpi tf_idf_mpi.c -lm
mpirun -np 4 ./tfidf_mpi doclist.txt out_mpi

### Serial version
cd ~/ACA-MPI-TFIDF/serial/
gcc -o tfidf_serial tf_idf_serial.c -lm
./tfidf_serial doclist.txt out_serial

------------------------------------------------------------
Developers
------------------------------------------------------------

Raffaele Cammi – Computer Engineering student at the University of Pavia.

This project was developed for the course *Advanced Computer Architecture* (A.Y. 2024–2025).

