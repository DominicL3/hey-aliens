# PSRchive Setup

PSRchive is is a library used to analyze pulsar data, which is pretty sweet. Unfortunately, for a Mac, PSRchive is a bit of a pain to set up. You'll need quite a few dependencies, and if you don't have them, setup will fail relentlessly.

Here are some tips if you're in need of them:

1. Clone the latest version of PSRchive by opening your Terminal and running the command  
```
git clone git://git.code.sf.net/p/psrchive/code psrchive
```

2. Set your environment variables. This part messed me up, so pay careful attention!

```
export PSRHOME=$HOME/Pulsar
export PATH=${PATH}:$PSRHOME/bin

export PGPLOT_DIR=$PSRHOME/pgplot
export PGPLOT_FONT=$PGPLOT_DIR/grfont.dat

export TEMPO2=$PSRHOME/tempo2
export PSRCAT_FILE=$PSRHOME/psrcat/psrcat.db
```
You can set the `PSRHOME` variable to wherever you installed the `psrchive` folder, at least that's where I set it to.

3. Dependencies galore! Make sure you have GNU Autotools updated by running
```
brew install autoconf automake libtool
```
4. Next, `cd` into the `psrchive` folder you downloaded and try running 
```
./bootstrap
./configure
```
If the bootstrap command fails, you probably need to check on your GNU autotools.

5. For me, my `./configure` command failed, throwing me a "Fortran 77" error. More specifically, it told me

```
Fortran 77 name-mangling scheme... configure: error: cannot compile a simple Fortran program
```
After some Googling, it turns out you need to install fortran too! Make sure you have Xcode and its command-line tools installed. You can then find the gfortran for Mac at https://github.com/fxcoudert/gfortran-for-macOS/releases

You can find more the actual PSRchive tips here at http://psrchive.sourceforge.net/download.shtml and of course, you can always talk to Uncle Google if you're in desperation.