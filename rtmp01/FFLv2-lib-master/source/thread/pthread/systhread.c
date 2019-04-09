#include <pthread.h>
#include <FFL_Core.h>

#if HAVE_PTHREAD_NP_H
#include <pthread_np.h>
#endif


#include <signal.h>

#ifdef __LINUX__
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif


#include <dlfcn.h>
#ifndef RTLD_DEFAULT
#define RTLD_DEFAULT NULL
#endif


#include "internal_thread_sys.h"
typedef struct FFL_Thread_sys FFL_Thread;



#ifndef __NACL__
/* List of signals to mask in the subthreads */
static const int sig_list[] =
        {
    SIGHUP, SIGINT, SIGQUIT, SIGPIPE, SIGALRM, SIGTERM, SIGCHLD, SIGWINCH,
    SIGVTALRM, SIGPROF, 0
};
#endif


typedef struct ThreadContext
{
    void *args;
    FFL_ThreadLoop fn;
}ThreadContext;


static void * RunThread(void *args)
{
    ThreadContext* context = (ThreadContext*)args;
    (context->fn)(context->args);

    FFL_free(context);
    return NULL;
}

#if defined(__MACOSX__) || defined(__IPHONEOS__)
static FFL_bool checked_setname = FFL_FALSE;
static int (*ppthread_setname_np)(const char*) = NULL;
#elif defined(__LINUX__)
static FFL_bool checked_setname = FFL_FALSE;
static int (*ppthread_setname_np)(pthread_t, const char*) = NULL;
#endif

int FFL_SYS_CreateThread(FFL_Thread * thread, void *args,FFL_ThreadLoop fn)
{
    pthread_attr_t type;

    /* do this here before any threads exist, so there's no race condition. */
    #if defined(__MACOSX__) || defined(__IPHONEOS__) || defined(__LINUX__)
    if (!checked_setname) {
        void *fn = dlsym(RTLD_DEFAULT, "pthread_setname_np");
        #if defined(__MACOSX__) || defined(__IPHONEOS__)
        ppthread_setname_np = (int(*)(const char*)) fn;
        #elif defined(__LINUX__)
        ppthread_setname_np = (int(*)(pthread_t, const char*)) fn;
        #endif
        checked_setname = SDL_TRUE;
    }
    #endif

    /* Set the thread attributes */
    if (pthread_attr_init(&type) != 0) {
        return FFL_set_error("Couldn't initialize pthread attributes");
    }
    pthread_attr_setdetachstate(&type, PTHREAD_CREATE_JOINABLE);
    
    /* Set caller-requested stack size. Otherwise: use the system default. */
    if (thread->stacksize) {
        pthread_attr_setstacksize(&type, (size_t) thread->stacksize);
    }

    ThreadContext* context=0;
    context= FFL_mallocz(sizeof( ThreadContext));
    context->args = args;
    context->fn = fn;

    /* Create the thread and go! */
    if (pthread_create(&thread->handle, &type, RunThread,context) != 0)
    {
        FFL_free(context);
        return FFL_set_error("Not enough resources to create thread");
    }

    return 0;
}

void FFL_SYS_SetThreadName(const char *name)
{
#if !defined(__NACL__)
    int i;
    sigset_t mask;
#endif /* !__NACL__ */

    if (name != NULL) {
        #if defined(ANDROID)
            pthread_setname_np(pthread_self(),name);
        #elif defined(__MACOSX__) || defined(__IPHONEOS__) || defined(__LINUX__)
        SDL_assert(checked_setname);
        if (ppthread_setname_np != NULL) {
            #if defined(__MACOSX__) || defined(__IPHONEOS__)
            ppthread_setname_np(name);
            #elif defined(__LINUX__)
            ppthread_setname_np(pthread_self(), name);
            #endif
        }
        #elif HAVE_PTHREAD_SETNAME_NP
            #if defined(__NETBSD__)
            pthread_setname_np(pthread_self(), "%s", name);
            #else
            pthread_setname_np(pthread_self(), name);
            #endif
        #elif HAVE_PTHREAD_SET_NAME_NP
            pthread_set_name_np(pthread_self(), name);
        #elif defined(__HAIKU__)
            /* The docs say the thread name can't be longer than B_OS_NAME_LENGTH. */
            char namebuf[B_OS_NAME_LENGTH];
            SDL_snprintf(namebuf, sizeof (namebuf), "%s", name);
            namebuf[sizeof (namebuf) - 1] = '\0';
            rename_thread(find_thread(NULL), namebuf);
        #endif
    }

   /* NativeClient does not yet support signals.*/
#if !defined(__NACL__)
    /* Mask asynchronous signals for this thread */
    sigemptyset(&mask);
    for (i = 0; sig_list[i]; ++i) {
        sigaddset(&mask, sig_list[i]);
    }
    pthread_sigmask(SIG_BLOCK, &mask, 0);
#endif /* !__NACL__ */


#ifdef PTHREAD_CANCEL_ASYNCHRONOUS
    /* Allow ourselves to be asynchronously cancelled */
    {
        int oldstate;
        pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldstate);
    }
#endif
}

FFL_ThreadID FFL_SYS_ThreadID(void)
{
    return ((FFL_ThreadID) pthread_self());
}

int FFL_SYS_SetThreadPriority(FFL_ThreadPriority priority)
{
#if __NACL__ 
    /* FIXME: Setting thread priority does not seem to be supported in NACL */
    return 0;
#elif __LINUX__
    int value;

    if (priority == SDL_THREAD_PRIORITY_LOW) {
        value = 19;
    } else if (priority == SDL_THREAD_PRIORITY_HIGH) {
        value = -20;
    } else {
        value = 0;
    }
    if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), value) < 0) {
        /* Note that this fails if you're trying to set high priority
           and you don't have root permission. BUT DON'T RUN AS ROOT!

           You can grant the ability to increase thread priority by
           running the following command on your application binary:
               sudo setcap 'cap_sys_nice=eip' <application>
         */
        return FFL_set_error("setpriority() failed");
    }
    return 0;
#else
    struct sched_param sched;
    int policy;
    pthread_t thread = pthread_self();

    if (pthread_getschedparam(thread, &policy, &sched) < 0) {
        return FFL_set_error("pthread_getschedparam() failed");
    }
    if (priority == FFL_THREAD_PRIORITY_LOW) {
        sched.sched_priority = sched_get_priority_min(policy);
    } else if (priority == FFL_THREAD_PRIORITY_HIGH) {
        sched.sched_priority = sched_get_priority_max(policy);
    } else {
        int min_priority = sched_get_priority_min(policy);
        int max_priority = sched_get_priority_max(policy);
        sched.sched_priority = (min_priority + (max_priority - min_priority) / 2);
    }
    if (pthread_setschedparam(thread, policy, &sched) < 0) {
        return FFL_set_error("pthread_setschedparam() failed");
    }
    return 0;
#endif /* linux */
}

void FFL_SYS_WaitThread(FFL_Thread  * thread)
{
    pthread_join(thread->handle, 0);
}

void FFL_SYS_DetachThread(FFL_Thread  * thread)
{
    pthread_detach(thread->handle);
}

