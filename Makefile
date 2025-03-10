# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021-2023, ByteDance Ltd. and/or its Affiliates
# Author: Yuanhan Liu <liuyuanhan.131@bytedance.com>

MAKEFLAGS += --no-print-directory

export TPA_VERSION   = 1.0-rc0
export BUILD_MODE   ?= release
export DPDK_VERSION ?= v20.11.3
export NIC_TYPE     ?= mlnx

include buildtools/vars.mk

SUBDIRS = lib src test tools tpad app

.PHONY: all install clean distclean $(SUBDIRS) dpdk gtags scan-build so summary static

all: summary $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

test tools app: src lib
lib src: dpdk
tpad: tools lib
tools app: static

dpdk:
	$(Q)bash ./buildtools/build-dpdk.sh

$(LIBTPA_SO): src lib
	$(Q)echo "  LD $(notdir $@)"
	$(Q)$(CC) -shared -o $@ -L$(DPDK_LD_PATH)			\
	     -Wl,--whole-archive					\
	     $(OBJ_ROOT)/src/tpa-core.a				\
	     $(OBJ_ROOT)/lib/tpa-lib.a				\
	     $(DPDK_LD_PATH)/librte_*.a					\
	     -Wl,--no-whole-archive					\
	     $(LDFLAGS)
	$(Q)bash ./buildtools/gen-pkg-config-file

so: $(LIBTPA_SO)

static: src lib
	$(Q)echo "  AR libtpa.a"
	$(Q)echo create $(LIBTPA_A)                >  ./tmp/tpa.mri
	$(Q)echo addlib $(OBJ_ROOT)/src/tpa-core.a >> ./tmp/tpa.mri
	$(Q)echo addlib $(OBJ_ROOT)/lib/tpa-lib.a  >> ./tmp/tpa.mri
	$(Q)for i in $(DPDK_LD_PATH)/librte_*.a; do \
		echo addlib $$i			   >> ./tmp/tpa.mri; \
	done
	$(Q)echo save 				   >> ./tmp/tpa.mri
	$(Q)ar -M < ./tmp/tpa.mri
	$(Q)bash ./buildtools/gen-pkg-config-file

summary: $(SUBDIRS) static
	$(Q)echo ":: built $(shell buildtools/get-ver.sh): mode=$(BUILD_MODE) dpdk=$(DPDK_VERSION)"

install: $(SUBDIRS) static
	$(Q)echo "  INSTALL -> $(INSTALL_ROOT)"
	$(Q)sudo mkdir -p $(INSTALL_ROOT)
	$(Q)buildtools/get-ver.sh | sudo tee $(INSTALL_ROOT)/version > /dev/null
	$(Q)sudo install $(BUILD_ROOT)/libtpa*         $(INSTALL_ROOT)
	$(Q)sudo install $(BIN_ROOT)/app/*             $(INSTALL_ROOT)
	$(Q)sudo install $(BIN_ROOT)/tpad/*            $(INSTALL_ROOT)
	$(Q)sudo install $(BIN_ROOT)/tools/*           $(INSTALL_ROOT)
	$(Q)sudo install $(SRC_ROOT)/include/api/*     $(INSTALL_ROOT)
	$(Q)sudo install $(BIN_ROOT)/tools/tpa         /usr/bin
	$(Q)sudo install $(BIN_ROOT)/app/*             /usr/bin
	$(Q)sudo install $(BUILD_ROOT)/libtpa.pc       /usr/share/pkgconfig

scan-build: clean
	scan-build make

gtags:
	git ls-files | gtags -f -

html:
	sphinx-build doc build/html

html_pub:
	sphinx-build doc /var/www/html/libtpa-doc

clean:
	$(Q)echo "  CLEAN"
	$(Q)rm -rf $(BIN_ROOT) $(OBJ_ROOT) $(LIBTPA_A) $(LIBTPA_SO)

distclean:
	$(Q)echo "  DISTCLEAN"
	$(Q)rm -rf $(BUILD_ROOT)
