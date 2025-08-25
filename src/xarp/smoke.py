import asyncio
import threading


async def foo():
    raise Exception('ops')


async def main_async():
    print('main_async')
    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    def consume():
        try:
            # raise Exception('ops')
            print('before future')
            foo_coroutine = foo()
            future = asyncio.run_coroutine_threadsafe(foo_coroutine, loop)
            print('future:', future.result())
            print('after future')
            done.set()
        except Exception as e:
            print('exception in thread')
        finally:
            done.set()

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    print('before done.wait()')
    await done.wait()
    print('************** after done.wait()')


if __name__ == '__main__':
    asyncio.run(main_async())
