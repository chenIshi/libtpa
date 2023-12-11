#!/usr/bin/ruby
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023, ByteDance Ltd. and/or its Affiliates
# Author: Yuanhan Liu <liuyuanhan.131@bytedance.com>

require 'yaml'

class ParamMatrix
	def initialize(str)
		@names  = []
		@values = []

		YAML.load(str).each { |k, v|
			@names.push  k
			@values.push v
		}
	end

	def each(&block)
		@curr = {}

		iterate @values, 0, &block
	end

private
	def deepcopy(o)
		Marshal.load(Marshal.dump(o))
	end

	def iterate(obj, depth, &block)
		if depth == obj.size
			yield deepcopy(@curr)
			return
		end

		obj[depth].each { |v|
			@curr[@names[depth]] = v
			iterate obj, depth + 1, &block
		}
	end
end

class DefaultParams
	def initialize(str)
		@default_params = YAML.load str
	end

	def each(&block)
		@default_params.each { |p|
			yield p
		}
	end
end

class TestShell
	def initialize(name, params, default_params, body)
		@name = name
		@default_params = default_params
		@params = params
		@user_defined_params = {}
		@body = body

		construct_shell
	end

	def desc
		ret = @name + '/'

		@params.each { |k, v|
			v = v.to_s.gsub ' ', '_'
			ret += "#{k}=#{v}__"
		}

		ret.chomp "__"
	end

	def to_s
		@shell
	end

	def save(script)
		File.open(script, "w") { |f|
			f.write @shell
		}
	end

	def exec
		puts ":: running #{@name} with #{@params}"
		return if ENV['DRYRUN']

		script = "./testshell.sh"
		save script

		if not system "bash #{script}"
			# TODO: formal log
			system "echo failed to run #{Dir.pwd} | grep '.*' --color=always"
			return false
		end

		true
	end

	def append_user_defined_param(param)
		param = [ param ] if param.class != Array

		param.each { |var|
			k, v = var.split '='
			@user_defined_params[k] = v
		}

		# refresh it
		construct_shell
	end

	attr_reader :name
	attr_reader :params
	attr_reader :default_params

private
	def shell_append(buf)
		@shell += buf
	end

	def shell_append_params(prompt, params)
		shell_append "# #{prompt}\n"

		params.each { |k, v|
			shell_append "#{k}='#{v}'\n"
		}
		shell_append "\n"
	end

	def construct_shell
		@shell = ""

		shell_append "#!/bin/bash\n"
		shell_append "# test script auto generated by Matrix Shell\n"
		shell_append "\n"

		shell_append_params "default params", @default_params
		shell_append_params "test params", @params
		shell_append_params "user defined params", @user_defined_params

		shell_append "echo '#{@name} => #{@params}'\n"

		shell_append @body
	end
end

class MatrixShell
	def initialize(file)
		@name = File.basename file, ".ms"

		@param_matrix = {}
		@default_params = {}
		@body = ""

		@files = []
		@root_source_file = file
		append_source_file file
		parse
	end

	attr_reader :name
	attr_reader :param_matrix
	attr_reader :default_params

	def each(&block)
		@param_matrix.each { |param|
			yield TestShell.new(@name, param, @default_params, @body)
		}
	end

	def each_matched(pattern, &block)
		@param_matrix.each { |param|
			shell = TestShell.new(@name, param, @default_params, @body)
			next if pattern && shell.desc !~ pattern

			yield shell
		}
	end

private
	def append_source_file(path)
		path = File.dirname(@root_source_file) + '/' + path if not File.exist? path

		@files.push File.open(path)
	end

	def gets
		return nil if @files.empty?

		line = @files[-1].gets
		if line == nil
			@files.pop
			line = gets
		end

		line
	end

	def fetch_until_end_of_block
		ret = ""

		loop {
			line = gets
			break if line == "end\n"

			ret += line
		}

		ret
	end

	def parse
		loop {
			line = gets
			break if not line

			case line
			when "params:\n"
				@param_matrix = ParamMatrix.new fetch_until_end_of_block
			when "default_params:\n"
				@default_params = DefaultParams.new fetch_until_end_of_block
			when /#include (.*)/
				append_source_file $1
			else
				@body += line
			end
		}
	end
end
