WARNING:
    mistuba 3.6.4
    mistuba 3.7.1 supporta Compute Capability 6.1
    mistuna 3.8.0 ??? non testato
    mistuba 3.9.0 richiede Compute Capability 7.5



Mitsuba documentation
    https://mitsuba.readthedocs.io/en/stable/

Scene Format
    https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html

Plugins
    https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html

    emitters
        https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html
        area

    bdrf
        https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html


Other resources
    https://benedikt-bitterli.me/resources/


Installazione LLVM
------------------
    (NON FUNZIONA ancora)

    dopo aver installato LLVM, impostare la variabile d'ambiente:
    
        DRJIT_LIBLLVM_PATH=D:\LLVM\LLVM-11.1.0\bin\LLVM-C.dll
    
    Nota: PATH COMPLETO di  LLVM-C.dll
    
    
    https://github.com/mitsuba-renderer/drjit/issues/196
    
