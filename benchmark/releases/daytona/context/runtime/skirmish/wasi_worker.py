"""Untrusted WASI guest. NO preopened directories, environment, network, or host imports.

Launched in a separate process; the trusted parent kills it on decision timeout.
Fuel is a per-game instruction budget, identical for every submitted bot.
"""
import sys
import wasmtime


def main(path, fuel):
    config = wasmtime.Config()
    config.consume_fuel = True
    engine = wasmtime.Engine(config)
    store = wasmtime.Store(engine)
    store.set_fuel(int(fuel))
    store.set_limits(memory_size=256 * 1024 * 1024, instances=4, tables=4)
    wasi = wasmtime.WasiConfig()
    wasi.inherit_stdin()
    wasi.inherit_stdout()
    # stderr deliberately not inherited: guest logging cannot fill a pipe or disk.
    store.set_wasi(wasi)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    module = wasmtime.Module.from_file(engine, path)
    instance = linker.instantiate(store, module)
    try:
        instance.exports(store)['_start'](store)
    except wasmtime.Trap as error:
        # Wasmtime 38's Python TrapCode enum does not expose the fuel code.
        if 'all fuel consumed' in str(error):
            raise SystemExit(42) from None
        raise


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
